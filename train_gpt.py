import inspect
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as neural_network
from torch.nn import functional as neural_network_functions


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class CausalSelfAttention(neural_network.Module):
    """Let each token gather information from itself and earlier tokens only."""

    def __init__(self, model_config):
        super().__init__()

        # Each embedding must split evenly into independent attention heads.
        # For example, an embedding dimension of 384 and 6 heads gives each
        # head a 64-number representation (384 / 6).
        assert (
            model_config.embedding_dimension
            % model_config.number_of_attention_heads
            == 0
        )

        # One linear layer produces the query, key, and value vectors together.
        # Its output is three times as wide because it contains all three sets
        # of vectors, concatenated along the final embedding dimension.
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

        # GPT-2 scales residual-branch projections at initialization to keep
        # activations well behaved as blocks accumulate.
        self.output_projection.NANOGPT_SCALE_INIT = 1

        # Store these dimensions because the forward pass needs them to split
        # the combined embedding dimension into separate attention heads.
        self.number_of_attention_heads = model_config.number_of_attention_heads
        self.embedding_dimension = model_config.embedding_dimension

    def forward(self, input_embeddings):
        """Return context-aware embeddings with the same shape as the input.

        `input_embeddings` has shape
        (batch_size, sequence_length, embedding_dimension).
        Each position may attend only to positions at or before itself, which
        prevents the model from seeing future tokens.
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
        attention_head_dimension = (
            embedding_dimension // self.number_of_attention_heads
        )

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
        # token after i. On supported hardware, PyTorch may select an optimized
        # attention kernel automatically.
        attention_head_outputs = (
            neural_network_functions.scaled_dot_product_attention(
                queries,
                keys,
                values,
                is_causal=True,
            )
        )

        # Put heads back beside one another to recover one full embedding per
        # token. `contiguous()` makes the transposed data layout suitable for
        # `view`, which only reinterprets existing memory rather than copying it.
        combined_attention_output = (
            attention_head_outputs.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, embedding_dimension)
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

        # Use the exact GELU formulation rather than the tanh approximation.
        # This changes the numerical behavior slightly but leaves the architecture
        # otherwise unchanged.
        self.gelu_activation = neural_network.GELU(approximate="none")

        self.contract_projection = neural_network.Linear(
            expanded_embedding_dimension,
            model_config.embedding_dimension,
        )

        self.contract_projection.NANOGPT_SCALE_INIT = 1

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
        normalized_attention_input = self.attention_layer_normalization(
            residual_stream
        )
        residual_stream = residual_stream + self.causal_self_attention(
            normalized_attention_input
        )

        normalized_feed_forward_input = self.feed_forward_layer_normalization(
            residual_stream
        )
        return residual_stream + self.feed_forward_network(
            normalized_feed_forward_input
        )


@dataclass
class GPTConfig:
    """Collect the hyperparameters that determine the model's architecture."""

    # The maximum number of tokens that the model can process at one time.
    context_window_size: int = 1024

    # The number of distinct token IDs in the embedding table and output
    # vocabulary.
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

        # Initialize before tying these weights so the language-model head and
        # input embedding table share one consistently initialized parameter.
        self.apply(self._init_weights)
        self.language_model_head.weight = self.transformer[
            "token_embedding_table"
        ].weight

    def _init_weights(self, module):
        """Initialize weights using the initialization scheme used by GPT-2."""
        if isinstance(module, neural_network.Linear):
            standard_deviation = 0.02

            if hasattr(module, "NANOGPT_SCALE_INIT"):
                standard_deviation *= (
                    2 * self.config.number_of_transformer_blocks
                ) ** -0.5

            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=standard_deviation,
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, neural_network.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, token_ids, target_token_ids=None):
        """Convert token IDs into vocabulary logits at every sequence position.

        `token_ids` has shape (batch_size, sequence_length).
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

        # PyTorch broadcasts the position embeddings across the batch dimension.
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

        # Compute the standard next-token language-modeling loss when targets
        # are supplied. `reshape` is safe even when the tensor is non-contiguous.
        loss = None
        if target_token_ids is not None:
            loss = neural_network_functions.cross_entropy(
                vocabulary_logits.reshape(-1, vocabulary_logits.size(-1)),
                target_token_ids.reshape(-1),
            )

        return vocabulary_logits, loss

    @classmethod
    def from_pretrained(cls, pretrained_model_name):
        """Create this GPT implementation and load weights from GPT-2.

        Hugging Face stores the same GPT-2 architecture with shorter module
        names. This method creates our clearly named version, then copies each
        compatible parameter from the downloaded checkpoint into it.
        """
        supported_model_names = {
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
        }
        assert pretrained_model_name in supported_model_names

        # Import here because Transformers is needed only when loading an
        # existing checkpoint, not when training our own model from scratch.
        from transformers import GPT2LMHeadModel

        print(
            f"Loading weights from pretrained GPT-2 model: "
            f"{pretrained_model_name}"
        )

        # GPT-2 model size determines the number of blocks, heads, and embedding
        # dimensions. All released GPT-2 checkpoints share this vocabulary and
        # maximum context-window size.
        model_architectures = {
            "gpt2": {
                "number_of_transformer_blocks": 12,
                "number_of_attention_heads": 12,
                "embedding_dimension": 768,
            },
            "gpt2-medium": {
                "number_of_transformer_blocks": 24,
                "number_of_attention_heads": 16,
                "embedding_dimension": 1024,
            },
            "gpt2-large": {
                "number_of_transformer_blocks": 36,
                "number_of_attention_heads": 20,
                "embedding_dimension": 1280,
            },
            "gpt2-xl": {
                "number_of_transformer_blocks": 48,
                "number_of_attention_heads": 25,
                "embedding_dimension": 1600,
            },
        }

        checkpoint_model_config = model_architectures[pretrained_model_name].copy()
        checkpoint_model_config["vocabulary_size"] = 50257
        checkpoint_model_config["context_window_size"] = 1024

        # First create our own GPT object. Its randomly initialized weights are
        # immediately replaced by the pretrained checkpoint values below.
        model_config = GPTConfig(**checkpoint_model_config)
        model = cls(model_config)
        local_state_dictionary = model.state_dict()

        # Download and construct Hugging Face's implementation of the requested
        # GPT-2 checkpoint, then retrieve all of its tensors by name.
        hugging_face_model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name
        )
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
            (
                ".attn.c_attn.",
                ".causal_self_attention.query_key_value_projection.",
            ),
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
        # differ. Checking this before copying makes the later error easier
        # to understand and prevents silently skipping a parameter.
        assert len(hugging_face_parameter_names) == len(local_state_dictionary), (
            f"Mismatched parameter counts: "
            f"{len(hugging_face_parameter_names)} != "
            f"{len(local_state_dictionary)}"
        )

        for hugging_face_parameter_name in hugging_face_parameter_names:
            local_parameter_name = hugging_face_parameter_name

            for source_name, destination_name in state_dictionary_name_replacements:
                local_parameter_name = local_parameter_name.replace(
                    source_name,
                    destination_name,
                )

            hugging_face_tensor = hugging_face_state_dictionary[
                hugging_face_parameter_name
            ]
            local_tensor = local_state_dictionary[local_parameter_name]

            if hugging_face_parameter_name.endswith(
                weight_suffixes_requiring_transpose
            ):
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

    def configure_optimizers(
        self,
        weight_decay,
        learning_rate,
        device_type,
        verbose=True,
    ):
        """Create AdamW with weight decay applied only to matrix parameters."""

        # Start with every trainable parameter, using its full hierarchical name
        # so the grouping logic is easy to inspect and debug.
        trainable_parameters = {
            parameter_name: parameter
            for parameter_name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

        # Weight decay is applied to matrices (weights used in learned linear
        # transformations and embeddings), but not to biases or normalization
        # parameters.
        decayed_parameters = [
            parameter
            for parameter in trainable_parameters.values()
            if parameter.dim() >= 2
        ]
        non_decayed_parameters = [
            parameter
            for parameter in trainable_parameters.values()
            if parameter.dim() < 2
        ]

        optimizer_parameter_groups = [
            {
                "params": decayed_parameters,
                "weight_decay": weight_decay,
            },
            {
                "params": non_decayed_parameters,
                "weight_decay": 0.0,
            },
        ]

        number_of_decayed_parameters = sum(
            parameter.numel() for parameter in decayed_parameters
        )
        number_of_non_decayed_parameters = sum(
            parameter.numel() for parameter in non_decayed_parameters
        )

        if verbose:
            print(
                f"num decayed parameter tensors: {len(decayed_parameters)}, "
                f"with {number_of_decayed_parameters:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: "
                f"{len(non_decayed_parameters)}, "
                f"with {number_of_non_decayed_parameters:,} parameters"
            )

        # Fused AdamW is useful on CUDA when PyTorch exposes it. MPS does not
        # use this CUDA-specific path, so it falls back to standard AdamW.
        fused_optimizer_available = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        use_fused_optimizer = (
            fused_optimizer_available and device_type == "cuda"
        )

        if verbose:
            print(f"using fused AdamW: {use_fused_optimizer}")

        optimizer_kwargs = {
            "lr": learning_rate,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }

        if fused_optimizer_available:
            optimizer_kwargs["fused"] = use_fused_optimizer

        return torch.optim.AdamW(
            optimizer_parameter_groups,
            **optimizer_kwargs,
        )


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


def load_tokens(filename):
    """Load a NumPy token array and convert it to PyTorch long token IDs."""
    token_array = np.load(filename)
    token_array = token_array.astype(np.int32)
    return torch.from_numpy(token_array.astype(np.int64, copy=False))


class DataLoaderLite:
    """Serve consecutive next-token batches from Tiny Shakespeare."""

    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        input_file_path = (
            Path(__file__).parent
            / "data"
            / "tinyshakespeare"
            / "input.txt"
        )

        encoder = tiktoken.get_encoding("gpt2")

        # Store the complete tokenized corpus once. `torch.from_numpy` avoids an
        # unnecessary intermediate tensor copy compared with torch.tensor.
        encoded_tokens = np.asarray(
            encoder.encode(input_file_path.read_text()),
            dtype=np.int64,
        )
        self.tokens = torch.from_numpy(encoded_tokens)

        minimum_required_tokens = batch_size * sequence_length + 1
        assert self.tokens.numel() >= minimum_required_tokens, (
            "The dataset is too small for one batch. Reduce batch_size or "
            "sequence_length."
        )

        print(
            f"loaded {self.tokens.numel()} tokens from "
            f"{input_file_path.name}"
        )

        self.reset()

    def reset(self):
        """Start reading the corpus again from its first token."""
        self.current_position = 0

    def next_batch(self):
        """Return one batch of input and next-token target sequences."""
        tokens_per_batch = self.batch_size * self.sequence_length

        if (
            self.current_position + tokens_per_batch + 1
            > self.tokens.numel()
        ):
            self.reset()

        token_buffer = self.tokens[
            self.current_position :
            self.current_position + tokens_per_batch + 1
        ]

        input_token_ids = token_buffer[:-1].view(
            self.batch_size,
            self.sequence_length,
        )
        target_token_ids = token_buffer[1:].view(
            self.batch_size,
            self.sequence_length,
        )

        self.current_position += tokens_per_batch

        return input_token_ids, target_token_ids


# -----------------------------------------------------------------------------
# HellaSwag evaluation helper
# -----------------------------------------------------------------------------


def get_most_likely_row(token_ids, completion_mask, logits):
    """Return the completion index with the lowest average completion loss."""

    # Evaluate the autoregressive loss at every token position. The first logit
    # predicts the second token, so logits and token IDs must be shifted by one.
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_token_ids = token_ids[..., 1:].contiguous()

    flattened_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flattened_token_ids = shifted_token_ids.view(-1)

    token_losses = neural_network_functions.cross_entropy(
        flattened_logits,
        flattened_token_ids,
        reduction="none",
    )
    token_losses = token_losses.view(token_ids.size(0), -1)

    # Shift the mask for the same reason as the logits and token IDs: the loss
    # at position i belongs to the token predicted at position i + 1.
    shifted_completion_mask = completion_mask[..., 1:].contiguous()

    # Keep losses only inside the completion region.
    masked_token_losses = token_losses * shifted_completion_mask

    # Average the completion loss independently for each candidate row.
    total_completion_loss = masked_token_losses.sum(dim=1)
    completion_token_count = shifted_completion_mask.sum(dim=1)

    average_completion_loss = (
        total_completion_loss / completion_token_count
    )

    # The completion with the lowest average negative log-likelihood is the
    # model's most likely completion.
    most_likely_completion_index = average_completion_loss.argmin().item()

    return most_likely_completion_index


# -----------------------------------------------------------------------------
# Apple Silicon / device helpers
# -----------------------------------------------------------------------------


def get_device():
    """Select CUDA, Apple Metal (MPS), or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize_device(device):
    """Wait for asynchronous device work to finish before timing it."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def probe_supported_mixed_precision_dtype(device, preferred_dtype=torch.bfloat16):
    """Return a mixed-precision dtype that actually works on this device.
 
    PyTorch's MPS backend has a history of supporting bfloat16 for some ops
    but throwing on others (particular matmul sizes, particular PyTorch/macOS
    combinations). Rather than assuming bf16 works, this runs one small
    representative operation (a matmul, since that's what has historically
    broken) under autocast and falls back to float16 if it raises.
    """
    if device.type not in {"mps", "cuda"}:
        return None
 
    def _try_dtype(candidate_dtype):
        try:
            probe_left = torch.randn(256, 256, device=device)
            probe_right = torch.randn(256, 256, device=device)
            with torch.autocast(device_type=device.type, dtype=candidate_dtype):
                probe_result = probe_left @ probe_right
            synchronize_device(device)
            # Force materialization; some MPS failures only surface once the
            # result is actually read back rather than at dispatch time.
            probe_result.float().sum().item()
            return True
        except Exception:
            return False
 
    if _try_dtype(preferred_dtype):
        return preferred_dtype
 
    print(
        f"{preferred_dtype} matmul under autocast failed on {device}; "
        f"falling back to float16 mixed precision"
    )
 
    if _try_dtype(torch.float16):
        return torch.float16
 
    print(
        f"float16 matmul under autocast also failed on {device}; "
        f"disabling mixed precision and using float32"
    )
    return None


# -----------------------------------------------------------------------------
# Local training
# -----------------------------------------------------------------------------


def train(
    number_of_training_steps=50,
    batch_size=8,
    sequence_length=256,
    learning_rate=3e-4,
    weight_decay=0.1,
    use_mixed_precision=False,
    use_torch_compile=False,
    gradient_accumulation_steps=2,
):
    """Train GPT locally with device-aware settings.
 
    The model architecture remains unchanged. MPS mixed precision and
    torch.compile are opt-in because support and performance can vary between
    PyTorch and macOS versions.
 
    `gradient_accumulation_steps` lets you simulate a larger batch size than
    fits in memory at once: `gradient_accumulation_steps` micro-batches of
    size `batch_size` are averaged together before each optimizer step, so
    the effective batch size is `batch_size * gradient_accumulation_steps`.
    """
    device = get_device()
    device_type = device.type
 
    print(f"using device: {device}")
 
    # Reproducible initialization. CUDA has its own random-number generator;
    # MPS uses PyTorch's general generator.
    torch.manual_seed(1337)
 
    if device_type == "cuda":
        torch.cuda.manual_seed(1337)
 
    # This can improve matrix multiplication performance without changing the
    # model's architecture. It is especially useful for larger matrix products.
    torch.set_float32_matmul_precision("high")
 
    train_loader = DataLoaderLite(
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
 
    model = GPT(GPTConfig()).to(device)
 
    # Keep float32 as the stable default. When mixed precision is requested,
    # probe which autocast dtype this specific device/PyTorch combination
    # actually supports rather than assuming bfloat16 or float16 will work.
    mixed_precision_dtype = None
    if use_mixed_precision:
        if device_type in {"mps", "cuda"}:
            mixed_precision_dtype = probe_supported_mixed_precision_dtype(device)
            if mixed_precision_dtype is not None:
                print(f"using {mixed_precision_dtype} mixed precision via autocast")
        else:
            print("mixed precision requested, but CPU training remains float32")
 
    if use_torch_compile:
        # Compilation is deliberately optional: MPS support can vary by PyTorch
        # release, and for a small local model the compilation overhead may
        # outweigh the speedup.
        print("compiling model with torch.compile...")
        model = torch.compile(model)
 
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        device_type=device_type,
    )
 
    model.train()
 
    for training_step in range(number_of_training_steps):
        step_start_time = time.perf_counter()
 
        optimizer.zero_grad(set_to_none=True)
 
        # Accumulate gradients over several micro-batches before stepping the
        # optimizer, so the effective batch size is larger than what fits in
        # memory at once. Each micro-batch's loss is divided by the number of
        # accumulation steps so the accumulated gradient matches what a single
        # large batch would have produced.
        #
        # The running loss is kept as a device tensor and only pulled to the
        # CPU once, after the loop. Calling `.item()` inside the loop would
        # force a GPU/CPU sync on every micro-batch, serializing work that
        # would otherwise pipeline - this matters even more on MPS than CUDA,
        # since per-sync overhead tends to be higher there.
        accumulated_loss = torch.zeros((), device=device)
 
        for _ in range(gradient_accumulation_steps):
            input_token_ids, target_token_ids = train_loader.next_batch()
 
            input_token_ids = input_token_ids.to(device)
            target_token_ids = target_token_ids.to(device)
 
            if mixed_precision_dtype is not None:
                with torch.autocast(
                    device_type=device_type,
                    dtype=mixed_precision_dtype,
                ):
                    vocabulary_logits, micro_batch_loss = model(
                        input_token_ids,
                        target_token_ids,
                    )
            else:
                vocabulary_logits, micro_batch_loss = model(
                    input_token_ids,
                    target_token_ids,
                )
 
            scaled_loss = micro_batch_loss / gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += micro_batch_loss.detach() / gradient_accumulation_steps
 
        optimizer.step()
 
        # MPS and CUDA execute operations asynchronously, so synchronize before
        # measuring elapsed time or reading the loss back to the CPU. CPU
        # execution needs no explicit synchronization.
        synchronize_device(device)
        accumulated_loss = accumulated_loss.item()
 
        step_end_time = time.perf_counter()
        step_duration_seconds = step_end_time - step_start_time
 
        processed_tokens = (
            train_loader.batch_size
            * train_loader.sequence_length
            * gradient_accumulation_steps
        )
        tokens_per_second = processed_tokens / step_duration_seconds
 
        print(
            f"step {training_step:04d}, "
            f"loss: {accumulated_loss:.6f}, "
            f"dt: {step_duration_seconds * 1000:.2f}ms, "
            f"tok/sec: {tokens_per_second:.2f}"
        )
 
    return model
 
 
if __name__ == "__main__":
    train()
