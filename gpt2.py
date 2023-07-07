import jax
import jax.numpy as np
from jax import random
from tqdm import tqdm
from utils import load_encoder_hparams_and_params


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x, temp=1):
    exp_x = np.exp((x - np.max(x, axis=-1, keepdims=True)) / temp)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(
        variance + eps
    )  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x


def attention(
    q, k, v, mask
):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    #     {'attn': {'c_attn': {'b':(2304,),'w':(768,2304)},
    #               'c_proj': {'b':(768,), 'w':(768, 768)} },
    n_head = 12

    # qkv projection
    # for Alan turing is a -> [5, 768] to [5, 2304]
    x = linear(x, c_attn["w"], c_attn["b"])  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    # [5, 2304] to [3, 5, 768]
    qkv = x.reshape(
        x.shape[0], 3, 768
    )  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd] | (3, n_seq, 768)
    # split into heads
    # [5, 3, 768] to [5, 3, 12, 64]
    qkv_heads = qkv.reshape(x.shape[0], 3, 12, 64)

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    # out_heads = [attention(qkv_heads[:, 0, i, :],
    #                       qkv_heads[:, 1, i, :],
    #                      qkv_heads[:, 2, i, :], causal_mask) for i in range(12)]
    # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]
    out_heads = jax.vmap(attention, in_axes=(1, 1, 1, None))(
        qkv_heads[:, 0, :, :], qkv_heads[:, 1, :, :], qkv_heads[:, 2, :, :], causal_mask
    )
    # merge heads
    #x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
    x = out_heads.transpose(1,0,2).reshape(x.shape[0], 64 * 12)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, c_proj["w"], c_proj["b"])  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2):  # [n_seq, n_embd] -> [n_seq, n_embd]
    #     {'attn': {'c_attn': {'b':(2304,),'w':(768,2304)},
    #               'c_proj': {'b':(768,), 'w':(768, 768)} },
    #      'ln_1':{'b':(768,), 'g':(768,)},
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    #      'mlp':{'c_fc': {'b': (3072,), 'w': (768, 3072)},
    #             'c_proj': {'b': (768,), 'w': (3072, 768)}}}]
    #      'ln_2':{'b':(768,), 'g':(768,)},
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    #           'wpe': (1024, 768)],
    #           'wte': (50357, 7681)}
    # token + positional embeddings
    x = (
        wte[np.array(inputs)] + wpe[np.arange(len(inputs))]
    )  # [n_seq] -> [n_seq, n_embd]

    #          {'blocks': [y,

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab

    #'ln_f': {'b': (768,), 'g': (768,)},
    x = layer_norm(x, ln_f["g"], ln_f["b"])  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def main(
    prompt: str = ":",
    n_tokens_to_generate: int = 100,
    model_size: str = "124M",
    models_dir: str = "models",
):

    prompt: str = """This is a conversation between Penny and Leonard.
    Penny: Hey leonard.
    Leonard: Hey penny, I love you.
    Penny:"""
    n_tokens_to_generate: int = 500
    model_size: str = "124M"
    models_dir: str = "models"
    print(f"Prompt: {prompt}")
    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    hparams = {
        "n_vocab": 50257,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
    }
    # params = {'blocks': [,
    #           'ln_f': {'b': (768,), 'g': (768,)},
    #           'wpe': (1024, 768)],
    #           'wte': (50357, 7681)}
    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)
    orig_len = len(input_ids)
    print(f"Input_ids: {input_ids}")
    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    key = random.PRNGKey(0) # seed

    # generate output ids
    #    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

#    for _ in tqdm(
#        range(n_tokens_to_generate), "generating"
#    ):  # auto-regressive decode loop
    last_backtrack = 10000
    for _  in range(n_tokens_to_generate):       
        max_backtrack = 2
        while True:
            key, subkey = random.split(key)

            logits = gpt2(
                np.array(input_ids), **params, n_head=12
            )  # model forward pass (n_sec, n_vocab)
              # greedy sampling, -1 for the last word
            soft_logits = softmax(logits[-1], temp=0.8)
            if np.sum(soft_logits > 0.1) < 2:
                print("greedy")
                next_id = np.argmax(logits[-1])
                #next_id = random.choice(subkey, np.arange(len(logits[-1])), p=np.sqrt(soft_logits))
            else:
                print("random")
                next_id = random.choice(subkey, np.arange(len(logits[-1])), p=soft_logits)

            token_p = soft_logits[next_id]
            if token_p < 0.5 and len(input_ids) >= orig_len and max_backtrack > 0 and last_backtrack > 2:
                print(f"Backtracking |{encoder.decode([int(next_id)])}| with p={token_p}")
                input_ids = input_ids[:-1]
                max_backtrack -= 1
                #n_tokens_to_generate += 1
                last_backtrack = 0
                continue

            input_ids.append(int(next_id))  # append prediction to input
            
            token_p = soft_logits[next_id]
            last_backtrack += 1
            print(f"Generating |{encoder.decode([int(next_id)])}| with p={token_p}")
            break
        if input_ids[-2:] == [20131, 25]:
            print(encoder.decode(input_ids))
            input_ids += [220] + encoder.encode(input())

    #output_ids = input_ids[
    #    len(input_ids) - n_tokens_to_generate :
    #]  # only return generated ids
    output_ids = input_ids
    print(f"output_ids: {output_ids}")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


def train():
    def lm_loss(params, inputs) -> float:
        import jax.nn

        x, y = inputs[:-1], inputs[1:]
        output = gpt2(x, **params, n_head=12)
        output_soft = softmax(output)
        loss = np.mean(-np.log(np.array([output_soft[i, y[i]] for i in range(len(y))])))
        return loss

    grads = jax.grad(lm_loss)(params, inputs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
