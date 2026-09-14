# minigpt

This will be a GPT-2-style transformer, coded from scratch largely following Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master). I will possibly fine-tune it and make some improvements on the UI and efficiency.

This is not trying to be something fancy, but merely a way for me to force myself to learn to inner workings of transformers.


## Sample output

After being trained on the tiny shakespeare dataset for 30 minutes, or around 4000 steps (and with batch size etc. as in the file), our model achieves a training loss of around 1.46, but a poor validation loss of 6.81. In other words, the model is overfitting to the dataset – I'm sure this could have been dealt with by an expert, even with my very limited hardware. 

![training loss chart](figures/training_loss.png)



With this setup and being prompted with "Oh, Romeo," our model returns:

```
Oh Romeo, he's lord, and on him;
With five thousand times won.

Romeoure men.

QUEEN ELIZABETH:
What else? peace--

KING RICHMOND with a horse!

QUEEN ELIZABETH:
Wash thou and his own wrath, in his love, in their lives, in our arms,
I do forefo,
That we are butcher'd a king, that black, life.

KING RICHARD III:
Why should I'll no other beauties.

KING RICHMakes me the field.
QUEEN ELIZABETH:
But you do good my son Edward still infect another.

KING RICHARD III:
My gracious sovereign account of heaven! there the town of Clarence, what is spake in France;
When come and the land, what, so, true, this your grace is.

MONTAGUEEN ELIZABETH:
What's hand, that will our cousin, when I'll inform'd
For nothing else you do you homely and tell him of patience
And thus I'll frighted with our side, to-morrow, my soul!

KING RICHARD III, and sovereign, for a king,
And hate I amends that hath twenty winters out:
Look, and bring me the king, what with a-morrow 'larhips?
He is your grace for we are!
RIVERS:
O God! a-day?

DUCHESS OF YORK:
O my lord, then, then, for loss of any be.

Tis love, what rests me, or be brief,
QUEEN ELIZABETH:
On what services are all's lords, as you.

QUEEN ELIZABETH:
I do not be thus?

KING RICHARD III:
'Twere trinicious way.

ARCHBISHOP OF YORK:
're you both, gentle words as part of you,
Even so?  what is, and being thus I have spent,
But I be tempted me the people.

DUCHESS OF YORK:
Alas I be patient: why, that, good for you had said,
An I have wrought us no;
And so we
```

## Changes to nanoGPT

The model is architecturally identical to Karpathy's nanoGPT: same pre-norm Transformer blocks, causal self-attention via `scaled_dot_product_attention`, GELU-based MLP, and weight tying between the token embedding and the output head. The differences are:

- **Fully spelled-out naming.** Every module, variable, and config field uses a descriptive name instead of nanoGPT's compact ones (`c_attn` → `query_key_value_projection`, `wte`/`wpe` → `token_embedding_table`/`position_embedding_table`, `n_embd` → `embedding_dimension`, etc.). The point of this project is to force myself to actually understand each piece, so I optimized the code for reading over terseness.
- **Apple Silicon support alongside CUDA.** Added `get_device()` and `synchronize_device()` helpers so the same training loop runs on MPS, CUDA, or CPU, since I'm developing on an M4 Air rather than an Nvidia GPU.
- **Defensive mixed-precision handling.** Rather than assuming bf16/fp16 autocast works, the training loop probes the current device with a small matmul before committing to a dtype, and falls back to float32 if neither works — MPS support for this is inconsistent across PyTorch versions. This ended up being a big bottleneck as MPS doesn't support bf16. I could have tried running it on the CPU with bf16, but I still think this would perform worse.


## Differences to "the" Transformer

On the left is the Transformer architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and on the right is ours.

Since this project follows GPT-2's architecture (via nanoGPT) rather than the original paper's, the main differences are:

- **Decoder-only, no encoder.** The original paper describes an encoder-decoder architecture for sequence-to-sequence tasks (translation). This project keeps only the decoder stack, since it's trained purely as a language model.
- **No cross-attention.** With no encoder, there's nothing for a decoder to attend to besides its own previous tokens — so each block has just one attention sublayer (causal self-attention) followed by the feed-forward sublayer, rather than the original's self-attention → cross-attention → feed-forward sequence.
- **Pre-normalization instead of post-normalization.** LayerNorm is applied *before* each sublayer (attention or feed-forward) here, with the sublayer's output then added to the residual stream. The original paper normalizes *after* the residual addition. Pre-norm is the choice GPT-2 made for training stability at depth.
- **Learned positional embeddings instead of fixed sinusoidal ones.** The original paper computes fixed sinusoidal position encodings. This project (like GPT-2) instead learns a position embedding table jointly with the rest of the model, at the cost of a hard maximum sequence length.
- **GELU instead of ReLU** in the feed-forward sublayer. 

![Comparison between this project's transformer block and the original "Attention Is All You Need" architecture](figures/transformer_comparison.png)