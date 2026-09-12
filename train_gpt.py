import os
import math
import time
import inspect
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as neural_network
from torch.nn import functional as neural_network_functions

 


class CausalSelfAttention(neural_network.Module):
    """Let each token gather information from itself and earlier tokens only."""

    def __init__(self, model_config):
        super().__init__()

        # Each embedding must split evenly into independent attention heads.
        # For example, an embedding dimension of 384 and 6 heads gives each
        # head a 64-number representation (384 / 6).
        assert model_config.embedding_dimension % model_config.number_of_attention_heads == 0

        # One linear layer produces the query, key, and value vectors together.
        # Its output is three times as wide because it contains all three sets
        # of vectors, concatenated along the final (embedding) dimension.
        self.query_key_value_projection = neural_network.Linear(
            model_config.embedding_dimension,
            3 * model_config.embedding_dimension,
        )

        # After attention heads are combined, this mixes their information back
        # into one embedding vector per token.
        self.output_projection = neural_network.Linear(
            model_config.embedding_dimension,
            model_config.embedding_dimension,
        )

        # Store these dimensions because the forward pass needs them to split
        # the combined embedding dimension into separate attention heads.
        self.number_of_attention_heads = model_config.number_of_attention_heads
        self.embedding_dimension = model_config.embedding_dimension

    def forward(self, input_embeddings):
        """Return context-aware embeddings with the same shape as the input.

        `input_embeddings` has shape (batch_size, sequence_length,
        embedding_dimension). Each position may attend only to positions at or
        before itself, which prevents the model from seeing future tokens.
        """
        batch_size, sequence_length, embedding_dimension = input_embeddings.size()

        # Project every input embedding into its query, key, and value vectors,
        # then split the final dimension into three equal-sized pieces.
        query_key_value_embeddings = self.query_key_value_projection(input_embeddings)
        queries, keys, values = query_key_value_embeddings.split(
            self.embedding_dimension,
            dim=2,
        )

        # Reshape from one embedding dimension into multiple heads. Transposing
        # produces (batch_size, number_of_heads, sequence_length, head_size),
        # which lets PyTorch perform attention independently for every head.
        attention_head_dimension = embedding_dimension // self.number_of_attention_heads
        keys = keys.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            attention_head_dimension,
        ).transpose(1, 2)
        queries = queries.view( 
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            attention_head_dimension,
        ).transpose(1, 2)
        values = values.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            attention_head_dimension,
        ).transpose(1, 2)

        # PyTorch calculates scaled dot-product attention. `is_causal=True`
        # applies the lower-triangular mask, so token i cannot attend to any
        # token after i. Newer PyTorch versions can use Flash Attention here.
        attention_head_outputs = neural_network_functions.scaled_dot_product_attention(
            queries,
            keys,
            values,
            is_causal=True,
        )

        # Put heads back beside one another to recover one full embedding per
        # token. `contiguous()` makes the transposed data layout suitable for
        # `view`, which only reinterprets existing memory rather than copying it.
        combined_attention_output = attention_head_outputs.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_dimension,
        )

        # A final learned projection allows information from different heads to
        # interact before this sublayer returns its result.
        return self.output_projection(combined_attention_output)


class FeedForwardNetwork(neural_network.Module):
    """Apply the per-token nonlinear transformation used in each GPT block."""

    def __init__(self, model_config):
        super().__init__()

        # GPT-2 expands each embedding to four times its width, applies GELU,
        # and then projects it back. Unlike attention, this treats each token
        # independently; communication between tokens happens in attention.
        expanded_embedding_dimension = 4 * model_config.embedding_dimension
        self.expand_projection = neural_network.Linear(
            model_config.embedding_dimension,
            expanded_embedding_dimension,
        )
        self.gelu_activation = neural_network.GELU(approximate="tanh")
        self.contract_projection = neural_network.Linear(
            expanded_embedding_dimension,
            model_config.embedding_dimension,
        )

    def forward(self, input_embeddings):
        """Transform every token embedding independently."""
        expanded_embeddings = self.expand_projection(input_embeddings)
        activated_embeddings = self.gelu_activation(expanded_embeddings)
        return self.contract_projection(activated_embeddings)


class TransformerBlock(neural_network.Module):
    """Combine causal attention and a feed-forward network with residual paths."""

    def __init__(self, model_config):
        super().__init__()

        # GPT-2 uses pre-normalization: normalize the input to each sublayer,
        # then add that sublayer's result to the unnormalized residual stream.
        self.attention_layer_normalization = neural_network.LayerNorm(
            model_config.embedding_dimension
        )
        self.causal_self_attention = CausalSelfAttention(model_config)
        self.feed_forward_layer_normalization = neural_network.LayerNorm(
            model_config.embedding_dimension
        )
        self.feed_forward_network = FeedForwardNetwork(model_config)

    def forward(self, residual_stream):
        """Update the residual stream once with attention and once with an MLP."""
        normalized_attention_input = self.attention_layer_normalization(residual_stream)
        residual_stream = residual_stream + self.causal_self_attention(normalized_attention_input)

        normalized_feed_forward_input = self.feed_forward_layer_normalization(residual_stream)
        return residual_stream + self.feed_forward_network(normalized_feed_forward_input)


@dataclass
class GPTConfig:
    """Collect the hyperparameters that determine the model's architecture."""

    # The maximum number of tokens that the model can process at one time.
    context_window_size: int = 1024
    # The number of distinct token IDs the embedding table and output predict.
    vocabulary_size: int = 50257
    # The number of Transformer blocks stacked one after another.
    number_of_transformer_blocks: int = 12
    # The number of separate attention patterns learned in every block.
    number_of_attention_heads: int = 12
    # The width of each token's internal vector representation.
    embedding_dimension: int = 768


class GPT(neural_network.Module):
    """Turn token IDs into next-token logits using a decoder-only Transformer."""

    def __init__(self, model_config):
        super().__init__()
        self.config = model_config

        # ModuleDict registers every child module with PyTorch, while descriptive
        # names make the model's three stages easy to identify in a state dict.
        self.transformer = neural_network.ModuleDict(
            {
                # Convert token IDs and positions into learned vectors. The
                # forward pass adds these two embedding types together.
                "token_embedding_table": neural_network.Embedding(
                    model_config.vocabulary_size,
                    model_config.embedding_dimension,
                ),
                "position_embedding_table": neural_network.Embedding(
                    model_config.context_window_size,
                    model_config.embedding_dimension,
                ),
                # Process the combined embeddings through the Transformer stack.
                "transformer_blocks": neural_network.ModuleList(
                    [
                        TransformerBlock(model_config)
                        for _ in range(model_config.number_of_transformer_blocks)
                    ]
                ),
                # Normalize once more before predicting the next token.
                "final_layer_normalization": neural_network.LayerNorm(
                    model_config.embedding_dimension
                ),
            }
        )

        # Map each final embedding to one unnormalized score (a logit) per token
        # in the vocabulary. Softmax is applied later when probabilities or loss
        # values are needed, rather than inside this layer.
        self.language_model_head = neural_network.Linear(
            model_config.embedding_dimension,
            model_config.vocabulary_size,
            bias=False,
        )

    def forward(self, token_ids, target_token_ids=None):
        """Convert token IDs into a vocabulary logit vector at every position.

        `token_ids` has shape (batch_size, sequence_length). The optional
        `target_token_ids` argument is reserved for the training-loss step that
        will be added later; this version of the method returns logits only.
        """
        batch_size, sequence_length = token_ids.size()

        # GPT-2 has a fixed-size learned position-embedding table, so a sequence
        # cannot be longer than the context window used to construct the model.
        assert sequence_length <= self.config.context_window_size, (
            f"Cannot process a sequence of length {sequence_length}; the context "
            f"window is only {self.config.context_window_size} tokens."
        )

        # Position indices are shared by every sequence in the batch: position
        # zero receives the same position embedding in every example, and so on.
        position_indices = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=token_ids.device,
        )

        # Token embeddings describe *which* token appears at each location.
        # Position embeddings describe *where* that token occurs in the sequence.
        token_embeddings = self.transformer["token_embedding_table"](token_ids)
        position_embeddings = self.transformer["position_embedding_table"](
            position_indices
        )

        # PyTorch broadcasts the position embeddings from
        # (sequence_length, embedding_dimension) across the batch dimension.
        residual_stream = token_embeddings + position_embeddings

        # Each Transformer block progressively updates the residual stream with
        # causal attention and then a per-token feed-forward transformation.
        for transformer_block in self.transformer["transformer_blocks"]:
            residual_stream = transformer_block(residual_stream)

        # Final normalization prepares the representations for the language-model
        # head, which produces one unnormalized vocabulary score per token.
        normalized_embeddings = self.transformer["final_layer_normalization"](
            residual_stream
        )
        vocabulary_logits = self.language_model_head(normalized_embeddings)

        # The returned tensor has shape
        # (batch_size, sequence_length, vocabulary_size).
        return vocabulary_logits

    
    @classmethod
    def from_pretrained(cls, pretrained_model_name):
        """Create this GPT implementation and load weights from a GPT-2 checkpoint.

        Hugging Face stores the same GPT-2 architecture with shorter module
        names. This method creates our clearly named version, then copies each
        compatible parameter from the downloaded checkpoint into it.
        """
        supported_model_names = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        assert pretrained_model_name in supported_model_names

        # Import here instead of at the top of the file because Transformers is
        # needed only when loading an existing checkpoint, not when training our
        # own model from scratch.
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT-2 model: {pretrained_model_name}")

        # GPT-2 model size determines the number of blocks, heads, and embedding
        # dimensions. All released GPT-2 checkpoints share this vocabulary and
        # maximum context-window size.
        model_architectures = {
            "gpt2": dict(number_of_transformer_blocks=12, number_of_attention_heads=12, embedding_dimension=768),
            "gpt2-medium": dict(number_of_transformer_blocks=24, number_of_attention_heads=16, embedding_dimension=1024),
            "gpt2-large": dict(number_of_transformer_blocks=36, number_of_attention_heads=20, embedding_dimension=1280),
            "gpt2-xl": dict(number_of_transformer_blocks=48, number_of_attention_heads=25, embedding_dimension=1600),
        }
        checkpoint_model_config = model_architectures[pretrained_model_name]
        checkpoint_model_config["vocabulary_size"] = 50257
        checkpoint_model_config["context_window_size"] = 1024

        # First create our own GPT object. Its randomly initialized weights are
        # immediately replaced by the pretrained checkpoint values below.
        model_config = GPTConfig(**checkpoint_model_config)
        model = cls(model_config)
        local_state_dictionary = model.state_dict()

        # Download and construct Hugging Face's implementation of the requested
        # GPT-2 checkpoint, then retrieve all of its tensors by name.
        hugging_face_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        hugging_face_state_dictionary = hugging_face_model.state_dict()

        # Hugging Face keeps two attention-mask buffers that are not trainable
        # parameters. Our implementation creates causal masks on demand through
        # `is_causal=True`, so these buffers have no corresponding local tensor.
        hugging_face_parameter_names = [
            parameter_name
            for parameter_name in hugging_face_state_dictionary
            if not parameter_name.endswith(".attn.masked_bias")
            and not parameter_name.endswith(".attn.bias")
        ]

        # These replacements translate Hugging Face's compact state-dict names
        # into the intentionally descriptive module names used in this project.
        state_dictionary_name_replacements = (
            ("transformer.wte.", "transformer.token_embedding_table."),
            ("transformer.wpe.", "transformer.position_embedding_table."),
            ("transformer.h.", "transformer.transformer_blocks."),
            (".ln_1.", ".attention_layer_normalization."),
            (".attn.c_attn.", ".causal_self_attention.query_key_value_projection."),
            (".attn.c_proj.", ".causal_self_attention.output_projection."),
            (".ln_2.", ".feed_forward_layer_normalization."),
            (".mlp.c_fc.", ".feed_forward_network.expand_projection."),
            (".mlp.c_proj.", ".feed_forward_network.contract_projection."),
            ("transformer.ln_f.", "transformer.final_layer_normalization."),
            ("lm_head.", "language_model_head."),
        )

        # OpenAI's original GPT-2 checkpoints use Conv1D layers for these
        # projections. PyTorch Linear layers store the same values with their
        # first two dimensions reversed, so these matrices need transposing.
        weight_suffixes_requiring_transpose = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        # A parameter-count mismatch usually means the two model architectures
        # differ. Checking this before copying makes the later error easier to
        # understand and prevents silently skipping a parameter.
        assert len(hugging_face_parameter_names) == len(local_state_dictionary), (
            f"Mismatched parameter counts: {len(hugging_face_parameter_names)} "
            f"!= {len(local_state_dictionary)}"
        )

        for hugging_face_parameter_name in hugging_face_parameter_names:
            local_parameter_name = hugging_face_parameter_name
            for source_name, destination_name in state_dictionary_name_replacements:
                local_parameter_name = local_parameter_name.replace(source_name, destination_name)

            hugging_face_tensor = hugging_face_state_dictionary[hugging_face_parameter_name]
            local_tensor = local_state_dictionary[local_parameter_name]

            if hugging_face_parameter_name.endswith(weight_suffixes_requiring_transpose):
                # Verify shapes before transposing, then copy without tracking
                # this assignment as part of PyTorch's gradient computation.
                assert hugging_face_tensor.shape[::-1] == local_tensor.shape
                with torch.no_grad():
                    local_tensor.copy_(hugging_face_tensor.t())
            else:
                # All other tensors use the same layout in both implementations.
                assert hugging_face_tensor.shape == local_tensor.shape
                with torch.no_grad():
                    local_tensor.copy_(hugging_face_tensor)

        return model
        

# --------------------------
"""Testing"""


model = GPT.from_pretrained('gpt2')
print("works so far!")
model.eval()
model.to('mps')

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(5, 1)
x = tokens.to('mps')

torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < 30:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = neural_network_functions.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)
        
for i in range(5):
    tokens = x[i, :30].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
