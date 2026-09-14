import inspect
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import matplotlib.pyplot as plt
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
    """Serve consecutive next-token batches from a tokenized text split."""

    def __init__(
        self,
        batch_size,
        sequence_length,
        split="train",
        validation_fraction=0.1,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.split = split

        input_file_path = Path(__file__).parent / "data" / "input.txt"
        encoder = tiktoken.get_encoding("gpt2")

        text = input_file_path.read_text(encoding="utf-8")
        encoded_tokens = np.asarray(
            encoder.encode(text),
            dtype=np.int64,
        )
        all_tokens = torch.from_numpy(encoded_tokens)

        split_index = int(all_tokens.numel() * (1.0 - validation_fraction))
        split_index = max(1, min(split_index, all_tokens.numel() - 1))

        if split == "train":
            self.tokens = all_tokens[:split_index]
        elif split in {"val", "validation"}:
            self.tokens = all_tokens[split_index:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        minimum_required_tokens = batch_size * sequence_length + 1
        assert self.tokens.numel() >= minimum_required_tokens, (
            f"The {split} split has only {self.tokens.numel()} tokens, but "
            f"{minimum_required_tokens} are required. Reduce batch_size or "
            f"sequence_length."
        )

        print(
            f"loaded {self.tokens.numel():,} {split} tokens from "
            f"{input_file_path.name}"
        )

        self.reset()

    def reset(self):
        """Start reading the current split again from its first token."""
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


@torch.no_grad()
def evaluate_loss(model, data_loader, device, number_of_batches=20):
    """Estimate average loss on a data split without changing model weights."""
    was_training = model.training
    model.eval()

    total_loss = 0.0

    for _ in range(number_of_batches):
        input_token_ids, target_token_ids = data_loader.next_batch()
        input_token_ids = input_token_ids.to(device)
        target_token_ids = target_token_ids.to(device)

        _, loss = model(input_token_ids, target_token_ids)
        total_loss += loss.item()

    if was_training:
        model.train()

    return total_loss / number_of_batches


def plot_losses(
    train_losses,
    validation_steps,
    validation_losses,
    output_path="training_loss.png",
):
    """Save a graph showing training and validation loss over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Training loss",
    )

    if validation_losses:
        plt.plot(
            validation_steps,
            validation_losses,
            marker="o",
            label="Validation loss",
        )

    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Tiny Shakespeare training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()


@torch.no_grad()
def generate_text(
    model,
    prompt,
    max_new_tokens=300,
    temperature=0.8,
    top_k=50,
):
    """Generate text autoregressively from a prompt."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    device = next(model.parameters()).device
    encoder = tiktoken.get_encoding("gpt2")

    token_ids = encoder.encode(prompt)
    if not token_ids:
        raise ValueError("prompt must contain at least one token")

    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    model.eval()

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.config.context_window_size:]

        logits, _ = model(context)
        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return encoder.decode(input_ids[0].tolist())



def train(
    training_minutes=30,
    batch_size=8,
    sequence_length=256,
    learning_rate=3e-4,
    weight_decay=0.1,
    gradient_accumulation_steps=2,
    validation_fraction=0.1,
    validation_interval=100,
    validation_batches=20,
):
    """Train the small GPT model on Tiny Shakespeare for a fixed amount of time."""
    device = get_device()
    device_type = device.type

    print(f"using device: {device}")

    torch.manual_seed(1337)

    if device_type == "cuda":
        torch.cuda.manual_seed(1337)

    torch.set_float32_matmul_precision("high")

    train_loader = DataLoaderLite(
        batch_size=batch_size,
        sequence_length=sequence_length,
        split="train",
        validation_fraction=validation_fraction,
    )

    validation_loader = DataLoaderLite(
        batch_size=batch_size,
        sequence_length=sequence_length,
        split="val",
        validation_fraction=validation_fraction,
    )

    # Small enough to train locally, but large enough to learn Shakespeare's
    # character/dialogue patterns reasonably well.
    model = GPT(
        GPTConfig(
            context_window_size=sequence_length,
            vocabulary_size=50257,
            number_of_transformer_blocks=4,
            number_of_attention_heads=4,
            embedding_dimension=256,
        )
    ).to(device)

    number_of_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"model parameters: {number_of_parameters:,}")

    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        device_type=device_type,
    )

    train_losses = []
    validation_steps = []
    validation_losses = []

    # The requested duration is a target, not an exact guarantee. The loop
    # finishes the current optimizer step before stopping.
    end_time = time.perf_counter() + training_minutes * 60
    training_step = 0

    model.train()

    while time.perf_counter() < end_time:
        training_step += 1
        step_start_time = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)

        for _ in range(gradient_accumulation_steps):
            input_token_ids, target_token_ids = train_loader.next_batch()

            input_token_ids = input_token_ids.to(device)
            target_token_ids = target_token_ids.to(device)

            _, micro_batch_loss = model(
                input_token_ids,
                target_token_ids,
            )

            (micro_batch_loss / gradient_accumulation_steps).backward()

            accumulated_loss += (
                micro_batch_loss.detach() / gradient_accumulation_steps
            )

        optimizer.step()

        synchronize_device(device)

        train_loss = accumulated_loss.item()
        train_losses.append(train_loss)

        step_duration_seconds = time.perf_counter() - step_start_time

        processed_tokens = (
            batch_size
            * sequence_length
            * gradient_accumulation_steps
        )
        tokens_per_second = processed_tokens / step_duration_seconds

        if (
            training_step == 1
            or training_step % validation_interval == 0
        ):
            validation_loss = evaluate_loss(
                model,
                validation_loader,
                device,
                number_of_batches=validation_batches,
            )
            validation_steps.append(training_step)
            validation_losses.append(validation_loss)

            elapsed_minutes = (
                (time.perf_counter() + training_minutes * 60 - end_time)
                / 60
            )

            print(
                f"step {training_step:04d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {validation_loss:.4f} | "
                f"{tokens_per_second:.0f} tok/s | "
                f"{elapsed_minutes:.1f} min"
            )
        else:
            print(
                f"step {training_step:04d} | "
                f"loss {train_loss:.4f} | "
                f"{tokens_per_second:.0f} tok/s"
            )

    print(f"\ntraining finished after {training_step} steps")

    plot_losses(
        train_losses,
        validation_steps,
        validation_losses,
        output_path="training_loss.png",
    )

    return model


if __name__ == "__main__":
    model = train(
        training_minutes=30,
        batch_size=8,
        sequence_length=256,
        learning_rate=3e-4,
        gradient_accumulation_steps=2,
        validation_interval=100,
        validation_batches=20,
    )

    generated_text = generate_text(
        model,
        prompt="Oh Romeo,",
        max_new_tokens=500,
        temperature=0.8,
        top_k=50,
    )

    print("\n" + "=" * 80)
    print("GENERATED TEXT")
    print("=" * 80)
    print(generated_text)
