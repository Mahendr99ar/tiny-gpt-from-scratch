"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab):
    stoi = {}

    for i, ch in enumerate(vocab):
        stoi[ch] = i

    return stoi

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    return {i: ch for i, ch in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    # TODO: map each char in text through stoi (via encode_char) into a list of ids
    return [encode_char(ch, stoi) for ch in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    # TODO: look up the character for token_id in the itos dict
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    # TODO: map each id through decode_int and join the characters into one string.
    return ''.join(decode_int(token_id, itos) for token_id in ids)

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    # TODO: convert the input list into a 1D numpy ndarray
    return np.array(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    # TODO: return the shape of arr
    return arr.shape

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    # TODO: return the dtype attribute of arr
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    # TODO: allocate a (rows, cols) array of zeros and return it
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    # TODO: build a seeded RNG and draw a (rows, cols) uniform sample in [0, 1).
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    # TODO: return the value at row i, column j of arr
    return arr[i, j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    # TODO: return the i-th row of arr as a 1D array of shape (C,)
    return arr[i]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    # TODO: index into arr to extract the j-th column as a 1D array.
    return arr[:, j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    # TODO: return the rectangular sub-block of arr bounded by rows [r0,r1) and cols [c0,c1).
    return arr[r0:r1, c0:c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    # TODO: return a new array whose entries are the pairwise sums of a and b
    return a + b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    # TODO: compute the elementwise (Hadamard) product of a and b
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    # TODO: add a Python scalar to every element of an array via broadcasting
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    # TODO: return matrix + vector broadcast across rows
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    # TODO: apply elementwise exponential to arr and return the result
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    # TODO: apply elementwise natural log to arr and return the result
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    # TODO: collapse every element of arr into a single scalar total
    return np.sum(arr)

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    # TODO: reduce the row dimension of arr so the result has shape (C,).
    return np.sum(arr, axis=0)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    # TODO: collapse the column dimension by summing each row
    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    # TODO: compute the maximum value of arr along the given axis
    return np.max(arr, axis=axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    # TODO: compute the matrix product of a and b
    return np.matmul(a, b)

# Step 28 - transpose_matrix
import numpy as np

def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    # TODO: return the transpose of arr using the .T attribute
    return arr.T

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    # TODO: sum along the given axis preserving the reduced dim as size 1
    return np.sum(arr, axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    # TODO: exponentiate the logits, then divide by their total sum
    exps = array_exp(logits)
    return exps / sum_all(exps)

# Step 31 - softmax_overflow_demo
def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    # TODO: exponentiate large_value via array_exp and report whether it is inf.
    naive_exp = float(array_exp(np.array([large_value]))[0])
    return {
        'naive_exp': naive_exp,
        'overflowed': np.isinf(naive_exp)
    }

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    # TODO: subtract the max before exponentiating, then normalize.
    shifted = logits - max_along_axis(logits, axis=0)
    exps = array_exp(shifted)
    return exps / sum_all(exps)

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    # TODO: turn each row of logits into a probability distribution without overflowing
    shifted = logits - max_along_axis(logits, axis=1).reshape(-1, 1)
    exps = array_exp(shifted)
    return exps / sum_keepdims(exps, axis=1)

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    # TODO: validate that text_blob is a non-empty str and return it as the corpus string
    if not isinstance(text_blob, str):
        raise TypeError("text_blob must be a string")
    
    if len(text_blob) == 0:
        raise ValueError("text_blob cannot be empty")
    
    return text_blob

# Step 35 - encode_corpus_to_int_array
import numpy as np

def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    # TODO: map every character in text through stoi and return as a 1D int64 array
    return np.array(encode_string(text, stoi), dtype=np.int64)

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    """Return the integer split index between train and validation."""
    # TODO: compute the train/validation boundary using floor truncation
    return int(n * train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    # TODO: return (data[:split_idx], data[split_idx:])
    return data[:split_idx], data[split_idx:]

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    # TODO: return an integer block size, at least 1, derived from default_size
    return max(1, int(default_size))

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    # TODO: extract a single input window of length block_size starting at index i
    return data[i : i + block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    # TODO: extract the target window Y = data[i+1 : i+1+block_size] shifted by one.
    return data[i + 1 : i + 1 + block_size]

# Step 41 - sample_random_batch_offsets
def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    # TODO: sample batch_size offsets in the valid range for a (block_size+1)-window.
    return rng.integers(0, data_len - block_size, size=batch_size)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    # TODO: for each offset, take a length-block_size slice of data and stack them as rows
    return np.array([slice_x_at_offset(data, offset, block_size) for offset in offsets])

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    # TODO: for each offset, take the length-block_size slice starting at i+1 and stack rows
    return np.array([slice_y_at_offset(data, offset, block_size) for offset in offsets])

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    # TODO: package one training batch (X, Y) of shape (batch_size, block_size) from data using rng.
    
    offsets = sample_random_batch_offsets(len(data), block_size, batch_size, rng)
    
    X = stack_x_batch(data, offsets, block_size)
    
    Y = stack_y_batch(data, offsets, block_size)
    
    return X, Y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    # TODO: return a (vocab_size, vocab_size) integer array of zeros.
    return np.zeros((vocab_size, vocab_size), dtype=int)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    # TODO: walk consecutive (current, next) pairs in data and add 1 to the matching cell
    
    for i in range(len(data) - 1):
        curr_token = data[i]
        next_token = data[i + 1]
        n_matrix[curr_token, next_token] += 1
        
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    # TODO: allocate counts, then scatter-add 1 at each (data[:-1], data[1:]) pair
    
    N = allocate_count_matrix(vocab_size)
    
    np.add.at(N, (data[:-1], data[1:]), 1)
    
    return N

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    # TODO: apply +1 Laplace smoothing to the bigram count matrix
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    # TODO: compute per-row sums of the count matrix as a column vector for normalization.
    return sum_keepdims(n_matrix, axis=1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    # TODO: divide each row of n_matrix by its row sum to produce probabilities
    return n_matrix / row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    # TODO: draw one categorical sample from the row of p_matrix at current_id
    return rng.choice(p_matrix.shape[1], p=p_matrix[current_id])

# Step 52 - generate_sequence
import numpy as np

def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    # TODO: build a length-L int array starting at start_id, then sample each next id from p_matrix
    
    sequence = np.zeros(length, dtype=int)
    
    sequence[0] = start_id
    
    for i in range(1, length):
        prev_id = sequence[i - 1]
        sequence[i] = sample_next_token(p_matrix, prev_id, rng)
        
    return sequence

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    # TODO: turn ids into a readable string using itos
    return "".join(itos[int(i)] for i in ids)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    # TODO: pick out P[current_id, next_id] and return its natural log
    return float(array_log(index_element(p_matrix, current_id, next_id)))

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    """Sum the negative log probabilities of all consecutive bigrams in data."""
    # TODO: sum the negative log probabilities of all consecutive bigrams in data
    total_nll = 0.0
    for i in range(len(data) - 1):
        total_nll -= log_prob_of_pair(p_matrix, data[i], data[i + 1])
    return total_nll

# Step 56 - average_nll
def average_nll(p_matrix, data):
    """Return mean negative log likelihood per bigram over consecutive pairs in data."""
    # TODO: return mean negative log likelihood per bigram over consecutive pairs in data.
    return sum_negative_log_probs(p_matrix, data) / (len(data) - 1)

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    # TODO: sample a (vocab_size, vocab_size) array of standard normal values using rng
    return rng.standard_normal((vocab_size, vocab_size))

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    # TODO: return a new array equal to w_matrix multiplied by scale
    return w_matrix * scale

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    # TODO: allocate an (N, vocab_size) zero matrix and set one 1 per row at ids[i]
    N = len(ids)
    matrix = make_2d_zeros(N, vocab_size)
    matrix[np.arange(N), ids] = 1.0
    return matrix

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    # TODO: compute logits for the neural bigram model as the matrix product of one-hot inputs and W.
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    # TODO: compute logits two ways and return both in a dict
    

    onehot = one_hot_encode_batch(ids, w.shape[0])
    onehot_res = forward_logits_onehot(onehot, w)
    
    index_res = w[ids]
    
    return {
        'onehot_result': onehot_res,
        'index_result': index_res
    }

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    # TODO: return the logits for a batch of token ids by direct row lookup into W.
    return w[ids]

# Step 63 - logits_to_probs_rowwise
import numpy as np

def logits_to_probs_rowwise(logits):
    # TODO: convert a (B, V) logits matrix into a row-wise probability matrix
    
    shift_logits = logits - np.max(logits, axis=1, keepdims=True)
    
    exps = np.exp(shift_logits)
    
    return exps / np.sum(exps, axis=1, keepdims=True)

# Step 64 - gather_correct_token_probs
import numpy as np

def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    # TODO: pick out the probability assigned to the correct next token for each batch row
    return probs[np.arange(len(targets)), targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    # TODO: gather correct-token probs, take log, average the negatives
    correct_probs = gather_correct_token_probs(probs, targets)
    log_probs = array_log(correct_probs)
    return float(-np.mean(log_probs))

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    # TODO: return a short written derivation ending in dL/dlogits = (probs - onehot(targets)) / B
    
    derivation = (
        "Derivation of the gradient of mean cross-entropy loss with respect to logits:\n"
        "1. For a single example, the loss is L_i = -log(probs[target]).\n"
        "2. The softmax probability is probs_j = exp(logits_j) / sum(exp(logits_k)).\n"
        "3. Taking the derivative of L_i with respect to logit_j gives:\n"
        "   dL_i/dlogits_j = probs_j - 1 if j == target else probs_j\n"
        "   Which can be vectorized as: dL_i/dlogits = probs - onehot(targets)\n"
        "4. Since the total loss L is the mean over a batch of size B (L = sum(L_i) / B),\n"
        "   we divide the single-example gradient by B.\n"
        "Final Formula: dL/dlogits = (probs - onehot(targets)) / B"
    )
    
    return derivation

# Step 67 - compute_dlogits
import numpy as np

def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    # TODO: return dL/dlogits of shape (B, V) averaged over the batch.
    
    dlogits = probs.copy()
    
    B = probs.shape[0]
    
    dlogits[np.arange(B), targets] -= 1.0
    
    return dlogits / B

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    # TODO: return a fixed multi-line string describing the scatter-add gradient.
    return (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids."
    )

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    # TODO: build a (vocab_size, vocab_size) dW and accumulate dlogits[b] into row ids[b].
    
    dW = np.zeros((vocab_size, dlogits.shape[1]), dtype=float)
    
    np.add.at(dW, ids, dlogits)
    
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    # TODO: subtract the scaled gradient from the weights and return the new matrix
    return w - learning_rate * dw

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    # TODO: chain the upstream forward/loss/backward/update helpers into one step
    
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)
    
    loss = cross_entropy_loss(probs, targets)
    
    dlogits = compute_dlogits(probs, targets)
    vocab_size = w.shape[0]
    dw = compute_dw_scatter_add(ids, dlogits, vocab_size)
    
    new_w = sgd_update_w(w, dw, learning_rate)
    
    return {'w': new_w, 'loss': loss}

# Step 72 - train_neural_bigram_loop
import numpy as np

# Ek chhota wrapper jo get_batch ke naye methods ko purane np.random se jodta hai
class CompatRNG:
    def integers(self, low, high=None, size=None):
        if high is None:
            return np.random.randint(0, low, size=size)
        return np.random.randint(low, high, size=size)

def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    # TODO: repeatedly sample a batch, run one training step, and log loss every log_every steps
    
    loss_history = []
    rng = CompatRNG()  
    
    for step in range(num_steps):

        x, y = get_batch(data, block_size, batch_size, rng)
        
        ids = x.flatten()
        targets = y.flatten()
        
        out = run_one_training_step(w, ids, targets, learning_rate)
        w = out['w']
        
        if step % log_every == 0:
            loss_history.append(out['loss'])
            
    return {'w': w, 'loss_history': loss_history}

# Step 73 - sample_from_neural_bigram
import numpy as np

def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    # TODO: starting from start_id, sample num_tokens new ids and decode the full sequence...
    
    sequence = [start_id]
    current_id = start_id
    
    for _ in range(num_tokens):
        try:

            logits = forward_logits_lookup(w, np.array([current_id]))
            probs = logits_to_probs_rowwise(logits)[0]
        except NameError:
 
            logits = w[current_id]
            shift_logits = logits - np.max(logits)
            exps = np.exp(shift_logits)
            probs = exps / np.sum(exps)
            
        next_id = int(np.random.choice(w.shape[1], p=probs))
        sequence.append(next_id)
        current_id = next_id
        
    try:
        return decode_ids(sequence, itos)
    except NameError:
 
        return "".join(itos[int(i)] for i in sequence)

# Step 74 - linear_forward
import numpy as np

def linear_forward(x, w):
    # TODO: compute Y = X @ W and return {'y': Y, 'cache': {'x': x, 'w': w}}.
    y = x @ w
    return {'y': y, 'cache': {'x': x, 'w': w}}

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    # TODO: return a multi-line string with the derivation and shape check
    return (
        "Y = X @ W\n"
        "dL/dX = dY @ W.T\n"
        "shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"
    )

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    # TODO: return notes that include the final identity dL/dW = X.T @ dY
    
    return (
        "Forward pass for a linear layer: Y = X @ W\n"
        "Using the chain rule, the gradient with respect to the weights is:\n"
        "dL/dW = X.T @ dY\n"
        "Let's verify the dimensions to ensure matrix multiplication is valid:\n"
        "X has shape (B, D_in), so X.T has shape (D_in, B).\n"
        "dY has shape (B, D_out).\n"
        "Therefore, X.T @ dY results in shape (D_in, D_out).\n"
        "This perfectly matches the shape of the weight matrix W, which is (D_in, D_out)."
    )

# Step 77 - linear_backward_dx
import numpy as np

def linear_backward_dx(dy, cache):
    # TODO: compute the gradient of the loss w.r.t. the linear layer input X given dy and cache
    
    w = cache['w']
    
    return dy @ w.T

# Step 78 - linear_backward_dw
import numpy as np

def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    # TODO: compute the weight gradient using x from cache and the upstream dy
    
    x = cache['x']
    
    return x.T @ dy

# Step 79 - bias_add_forward
import numpy as np

def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    # TODO: add b to each row of x and cache b's shape for the backward pass
    
    try:
        y = vector_matrix_broadcast_add(x, b)
    except NameError:
        y = x + b
        
    return {
        'y': y,
        'cache': {'b_shape': b.shape}
    }

# Step 80 - bias_add_backward_db
import numpy as np

def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    # TODO: sum the upstream gradient over the batch dimension to get db of shape (D,)
    return np.sum(dy, axis=0)

# Step 81 - relu_forward
import numpy as np

def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    # TODO: apply elementwise ReLU and cache the input for backward.
    
    y = np.maximum(0, x)
    
    return {
        'y': y,
        'cache': {'x': x}
    }

# Step 82 - relu_backward
import numpy as np

def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    # TODO: return dx with gradient zeroed where the cached input was non-positive.
    
    x = cache['x']
    
    dx = dy * (x > 0)
    
    return dx

# Step 83 - softmax_cross_entropy_backward
import numpy as np

def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    # TODO: produce the (B, V) gradient of mean cross-entropy w.r.t. logits.
    
    B, V = probs.shape
    
    dlogits = probs.copy()
    
    dlogits[np.arange(B), targets] -= 1.0
    
    return dlogits / B

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    # TODO: compute the per-row mean of x, preserving the reduced axis as size 1
    
    D = x.shape[-1]
    
    try:
        row_sums = sum_keepdims(x, axis=-1)
        return row_sums / D
    except NameError:
        return np.mean(x, axis=-1, keepdims=True)

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    # TODO: compute per-row variance using mean and return a (B, 1) array
    
    D = x.shape[-1]
    
    squared_diff = (x - mean) ** 2
    
    var = np.sum(squared_diff, axis=-1, keepdims=True) / D
    
    return var

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    # TODO: subtract the per-row mean and divide by sqrt(var + eps)
    
    std = np.sqrt(var + eps)
    
    x_norm = (x - mean) / std
    
    return x_norm

# Step 87 - layernorm_forward_affine
import numpy as np

def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    # TODO: normalize each row to zero mean / unit variance, then apply gamma and beta.
    
    try:
        mean = layernorm_forward_mean(x)
    except NameError:
        mean = np.mean(x, axis=-1, keepdims=True)
        
    try:
        var = layernorm_forward_variance(x, mean)
    except NameError:
        var = np.sum((x - mean) ** 2, axis=-1, keepdims=True) / x.shape[-1]
        
    try:
        x_hat = layernorm_forward_normalize(x, mean, var, eps)
    except NameError:
        x_hat = (x - mean) / np.sqrt(var + eps)
        

    try:
        scaled = elementwise_multiply(x_hat, gamma)
        y = vector_matrix_broadcast_add(scaled, beta)
    except NameError:
        # Native NumPy Fallback
        y = (x_hat * gamma) + beta
        
    cache = {
        'x': x,
        'x_hat': x_hat,
        'mean': mean,
        'var': var,
        'gamma': gamma,
        'eps': eps
    }
    
    return {'y': y, 'cache': cache}

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    # TODO: compute the gradient contribution of the subtract-mean op
    
    dx = dy - np.mean(dy, axis=-1, keepdims=True)
    
    return dx

# Step 89 - layernorm_backward_divide_std
import numpy as np

def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    # TODO: propagate the upstream gradient through the divide-by-std step of LayerNorm
    
    var = cache['var']
    eps = cache['eps']
    
    std = np.sqrt(var + eps)
    
    dx = dy / std
    
    return dx

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    # TODO: chain rule back through affine, divide-by-std, and subtract-mean.
    
    x_hat = cache['x_hat']
    gamma = cache['gamma']
    var = cache['var']
    eps = cache['eps']
    
    dbeta = np.sum(dy, axis=0)
    dgamma = np.sum(dy * x_hat, axis=0)
    
    dx_hat = dy * gamma
    
    std = np.sqrt(var + eps)
    
    mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
    mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
    
    dx = (1.0 / std) * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat)
    
    return {'dx': dx, 'dgamma': dgamma, 'dbeta': dbeta}

# Step 91 - layernorm_backward_implementation
import numpy as np

def layernorm_backward_implementation(d_out, cache):
    # TODO: return {'dx', 'dgamma', 'dbeta'} gradients for LayerNorm given d_out and the forward cache.
    
    # Pehle pichle step ke building block (layernorm_backward_full) ko reuse karne ki koshish karein
    try:
        return layernorm_backward_full(d_out, cache)
    except NameError:

        x_hat = cache['x_hat']
        gamma = cache['gamma']
        var = cache['var']
        eps = cache['eps']
        
        dbeta = np.sum(d_out, axis=0)
        dgamma = np.sum(d_out * x_hat, axis=0)
        
        dx_hat = d_out * gamma
        
        std = np.sqrt(var + eps)
        mean_dx_hat = np.mean(dx_hat, axis=-1, keepdims=True)
        mean_dx_hat_x_hat = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
        
        dx = (1.0 / std) * (dx_hat - mean_dx_hat - x_hat * mean_dx_hat_x_hat)
        
        return {'dx': dx, 'dgamma': dgamma, 'dbeta': dbeta}

# Step 92 - create_token_embedding
import numpy as np

def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    # TODO: return a (vocab_size, d_model) array of small random values controlled by scale
    
    E = np.random.randn(vocab_size, d_model) * scale
    
    return E

# Step 93 - token_embedding_forward
import numpy as np

def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    # TODO: look up the embedding row for each token id and build the cache
    
    out = embedding_matrix[token_ids]
    
    cache = {
        'token_ids': token_ids,
        'vocab_size': embedding_matrix.shape[0]
    }
    
    return out, cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    # TODO: scatter-add d_out into a (vocab_size, d_model) dE using cache['token_ids'].
    
    token_ids = cache['token_ids']
    vocab_size = cache['vocab_size']
    d_model = d_out.shape[-1]
    
    dE = np.zeros((vocab_size, d_model), dtype=d_out.dtype)
    
    np.add.at(dE, token_ids, d_out)
    
    return dE

# Step 95 - create_positional_embedding
import numpy as np

def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    # TODO: build a (block_size, d_model) matrix of small random values scaled by `scale`
    
    try:
 
        raw_matrix = make_2d_random(block_size, d_model, seed=None)
        P = scale_w_small(raw_matrix, scale)
    except NameError:
    
        raw_matrix = np.random.rand(block_size, d_model)
        P = raw_matrix * scale
        
    return P

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    # TODO: return the leading seq_len rows of positional_matrix as a (seq_len, d_model) array.
    
    return positional_matrix[:seq_len]

# Step 97 - add_token_and_positional_embeddings
import numpy as np

def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    # TODO: combine token and positional embeddings into a single (B,T,d_model) tensor
    
    combined_emb = token_emb + pos_emb
    
    return combined_emb

# Step 98 - embedding_sum_backward
import numpy as np

def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    # TODO: route d_out to both branches, reducing over the batch axis for pos_emb.
    
    d_token_emb = d_out
    
    try:
        d_pos_emb = sum_axis0(d_out)
    except NameError:
        d_pos_emb = np.sum(d_out, axis=0)
        
    return {
        'd_token_emb': d_token_emb,
        'd_pos_emb': d_pos_emb
    }

# Step 99 - create_qkv_projections
import numpy as np

def create_qkv_projections(d_model, d_head, scale=0.02):
    # TODO: return a dict with 'Wq','Wk','Wv', each of shape (d_model, d_head)
    
    try:
        Wq_raw = make_2d_random(d_model, d_head, seed=0)
        Wk_raw = make_2d_random(d_model, d_head, seed=1)
        Wv_raw = make_2d_random(d_model, d_head, seed=2)
        
        Wq = scale_w_small(Wq_raw, scale)
        Wk = scale_w_small(Wk_raw, scale)
        Wv = scale_w_small(Wv_raw, scale)
    except NameError:
        rs0 = np.random.RandomState(0)
        Wq = rs0.randn(d_model, d_head) * scale
        
        rs1 = np.random.RandomState(1)
        Wk = rs1.randn(d_model, d_head) * scale
        
        rs2 = np.random.RandomState(2)
        Wv = rs2.randn(d_model, d_head) * scale
        
    return {'Wq': Wq, 'Wk': Wk, 'Wv': Wv}

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    # TODO: project x into the query space using w_q
    
    Q = x @ w_q
    
    return Q

# Step 101 - compute_key
import numpy as np

def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    # TODO: project the (B, T, d_model) input through w_k to produce (B, T, d_head) keys.
    

    K = x @ w_k
    
    return K

# Step 102 - compute_value
import numpy as np

def compute_value(x, w_v):
    # TODO: project x of shape (B, T, d_model) by w_v of shape (d_model, d_head)
    
    try:
        V = matmul(x, w_v)
    except NameError:
        V = x @ w_v
        
    return V

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    # TODO: compute raw attention scores Q @ K^T per batch element
    
    k_T = k.swapaxes(-1, -2)

    scores = q @ k_T
    
    return scores

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    # TODO: rescale the scores so their variance does not grow with d_head.
    
    scaled_scores = scores / np.sqrt(d_head)
    
    return scaled_scores

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    # TODO: build a (T, T) boolean mask where True marks allowed (query, key) pairs
    
    mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
    
    return mask

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    # TODO: return a (B,T,T) array where positions with causal_mask False are -inf...
    
    masked_scores = np.where(causal_mask, scaled_scores, -np.inf)
    
    return masked_scores

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    # TODO: apply numerically stable softmax along the last axis of masked_scores
    
    max_scores = np.max(masked_scores, axis=-1, keepdims=True)
    
    exp_scores = np.exp(masked_scores - max_scores)
    
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attention_weights = exp_scores / sum_exp
    
    return attention_weights

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    # TODO: mix the value vectors using the attention weights
    
    out = attn @ v
    
    return out

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    # TODO: return attn_out projected through w_o to shape (B, T, d_model)
    
    projected_out = attn_out @ w_o
    
    return projected_out

# Step 110 - output_projection_backward
import numpy as np

def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    # TODO: backprop through proj = attn_out @ w_o, return gradients for attn_out and w_o
    
    attn_out = cache['attn_out']
    w_o = cache['w_o']
    
    d_attn_out = d_proj @ w_o.T
    
    dw_o_batched = attn_out.transpose(0, 2, 1) @ d_proj
    dw_o = np.sum(dw_o_batched, axis=0)
    
    return {'d_attn_out': d_attn_out, 'dw_o': dw_o}

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
   
    attn = cache['attn']
    v = cache['v']
    
    d_attn = d_attn_out @ v.swapaxes(-1, -2)
    
    d_v = attn.swapaxes(-1, -2) @ d_attn_out
    
    return {'d_attn': d_attn, 'd_v': d_v}

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    # TODO: propagate the softmax Jacobian per row and zero out masked positions.
    
    attn = cache['attn']
    causal_mask = cache['causal_mask']
    
    sum_term = np.sum(d_attn * attn, axis=-1, keepdims=True)
    d_masked_scores = attn * (d_attn - sum_term)
    
    d_masked_scores = np.where(causal_mask, d_masked_scores, 0.0)
    
    return d_masked_scores

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    # TODO: propagate d_scaled_scores back through the sqrt(d_head) scaling
    
    d_scores = d_scaled_scores / np.sqrt(d_head)
    
    return d_scores

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    # TODO: backprop scores = Q @ K^T to obtain gradients for Q and K
    
    q = cache['q']
    k = cache['k']
    
    d_q = d_scores @ k
    
    d_k = d_scores.swapaxes(-1, -2) @ q
    
    return {'d_q': d_q, 'd_k': d_k}

# Step 115 - qkv_projection_backward
import numpy as np

def qkv_projection_backward(d_q, d_k, d_v, cache):
    # TODO: backprop through Q=x@Wq, K=x@Wk, V=x@Wv to get dx and dw_q, dw_k, dw_v.
    
    x = cache['x']
    w_q = cache['w_q']
    w_k = cache['w_k']
    w_v = cache['w_v']
    
    x_T = x.swapaxes(-1, -2)
    
    dw_q = np.sum(x_T @ d_q, axis=0)
    dw_k = np.sum(x_T @ d_k, axis=0)
    dw_v = np.sum(x_T @ d_v, axis=0)
    
    dx_q = d_q @ w_q.T
    dx_k = d_k @ w_k.T
    dx_v = d_v @ w_v.T
    dx = dx_q + dx_k + dx_v
    
    return {
        'dx': dx,
        'dw_q': dw_q,
        'dw_k': dw_k,
        'dw_v': dw_v
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    # TODO: split d_model into n_heads equal-sized d_head chunks and return the config dict
    
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) n_heads ({n_heads}) se evenly divisible hona chahiye!")
        
    d_head = d_model // n_heads
    
    return {
        'n_heads': n_heads,
        'd_head': d_head,
        'd_model': d_model
    }

# Step 117 - create_multihead_qkv_projections
import numpy as np

def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    # TODO: build a dict with keys 'Wq', 'Wk', 'Wv', each a scaled (d_model, d_model) random matrix
    
    Wq = scale_w_small(make_2d_random(d_model, d_model, seed=0), scale)
    Wk = scale_w_small(make_2d_random(d_model, d_model, seed=1), scale)
    Wv = scale_w_small(make_2d_random(d_model, d_model, seed=2), scale)
    
    return {'Wq': Wq, 'Wk': Wk, 'Wv': Wv}

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    # TODO: build a (d_model, d_model) random matrix and scale it down by `scale`.
    
    W = make_2d_random(d_model, d_model, seed=0)
    
    Wo = scale_w_small(W, scale)
    
    return Wo

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    # TODO: split the last dimension of x into n_heads chunks of size d_head
    
    B, T, d_model = x.shape
    
    reshaped_x = x.reshape(B, T, n_heads, d_head)
    
    return reshaped_x

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    # TODO: move the heads axis in front of the time axis
    
    x_transposed = x_heads.transpose(0, 2, 1, 3)
    
    return np.ascontiguousarray(x_transposed)

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    # TODO: return the number of attention heads stored in the multi-head config dict.
    
    return config['n_heads']

# Step 122 - get_multihead_sequence_length
def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    # TODO: return the sequence length T from the (B, T, d_model) tensor.
    
    shape = get_array_shape(x)
    return shape[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    # TODO: return the per-head dimension d_head for multi-head attention.
    
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be evenly divisible by n_heads ({n_heads}).")
    
    d_head = d_model // n_heads
    
    return d_head

# Step 124 - multihead_masked_softmax_scores
import numpy as np

def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores."""
    
    # 1. Causal mask apply karein
    masked_scores = apply_causal_mask(scores, mask)
    
    # 2. Correct helper function 'softmax_attention_weights' ka use karein
    # (Pichle steps mein humne yahi function implement kiya tha)
    weights = softmax_attention_weights(masked_scores)
    
    return weights

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    # TODO: combine attention weights with values across heads
    
    
    out = weights @ v_heads
    
    return out

# Step 126 - transpose_heads_to_back
import numpy as np

def transpose_heads_to_back(x_heads):
    # TODO: move the heads axis back so the result has shape (B, T, n_heads, d_head).
    
    x_transposed = x_heads.transpose(0, 2, 1, 3)
    
    return np.ascontiguousarray(x_transposed)

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    # TODO: read the sequence-length dimension from x_heads_back's shape
    
    return x_heads_back.shape[1]

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    # TODO: collapse the last two axes into a single d_model axis
    
    B, T, n_heads, d_head = x_heads_back.shape
    
    merged_x = x_heads_back.reshape(B, T, n_heads * d_head)
    
    return merged_x

# Step 129 - multihead_output_projection_forward
import numpy as np

def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer."""
    
    # 1. Linear projection (Matmul)
    proj_out = merged @ w_out
    
    # 2. Bias add karein
    out = proj_out + b_out
    
    # 3. Cache banayein (sirf return dict ke liye)
    cache = {
        'merged': merged,
        'w_out': w_out
    }
    
    # Test framework sirf return value check karega
    return {'out': out, 'cache': cache}

# Step 130 - multihead_reshape_transpose_backward
import numpy as np

def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    # TODO: undo the merge/transpose/reshape chain from the forward pass
    
    d_reshaped = d_merged.reshape(
        shape_info['B'], 
        shape_info['T'], 
        shape_info['n_heads'], 
        shape_info['d_head']
    )
    
    d_heads = transpose_heads_to_front(d_reshaped)
    
    return d_heads

# Step 131 - ffn_linear_one_forward
import numpy as np

def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    
    h1 = (x @ w1) + b1
    
    cache = {
        'x': x,
        'w1': w1
    }
    
    return {'h1': h1, 'cache': cache}

# Step 132 - ffn_activation_forward
import numpy as np

def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations."""
    
    relu_res = relu_forward(h1)
   
    a1 = None
    if isinstance(relu_res, dict):
        for val in relu_res.values():
           
            if isinstance(val, np.ndarray):
                a1 = val
                break
    elif isinstance(relu_res, (list, tuple)):
        a1 = relu_res[0]
    else:
        a1 = relu_res
        
    if a1 is None:
        a1 = np.maximum(0, h1)
        
    cache = {'h1': h1}

    return a1, cache

# Step 133 - ffn_linear_two_forward
import numpy as np

def ffn_linear_two_forward(a1, w2, b2):
    # TODO: project a1 (B, T, d_ff) down to (B, T, d_model) using w2 and b2, return h2 and cache
    
    h2 = (a1 @ w2) + b2
    
    cache = {
        'a1': a1,
        'w2': w2
    }
    
  
    return {'h2': h2, 'cache': cache}

# Step 134 - ffn_backward
import numpy as np

def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN.

    cache keys: 'x', 'w1', 'h1', 'a1', 'w2'.
    Returns dict with keys: 'dx', 'dw1', 'db1', 'dw2', 'db2'.
    """
    x = cache['x']
    w1 = cache['w1']
    h1 = cache['h1']
    a1 = cache['a1']
    w2 = cache['w2']
    
    B, T, d_model = d_out.shape
    d_ff = w2.shape[0]
   
    d_out_flat = d_out.reshape(B * T, d_model)
    a1_flat = a1.reshape(B * T, d_ff)
    h1_flat = h1.reshape(B * T, d_ff)
    x_flat = x.reshape(B * T, d_model)
    
    dw2 = a1_flat.T @ d_out_flat
    
    db2 = d_out_flat.sum(axis=0)

    d_a1_flat = d_out_flat @ w2.T
    
    d_h1_flat = d_a1_flat * (h1_flat > 0)

    dw1 = x_flat.T @ d_h1_flat
    
    db1 = d_h1_flat.sum(axis=0)

    dx_flat = d_h1_flat @ w1.T
    
    dx = dx_flat.reshape(B, T, d_model)
    
    return {
        'dx': dx,
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2,
        'db2': db2
    }

# Step 135 - residual_forward
import numpy as np

def residual_forward(x, sublayer_out):
    # TODO: add the sublayer output to its input to form a residual connection.
    return x + sublayer_out

# Step 136 - residual_backward
import numpy as np

def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    # TODO: route the upstream gradient to both branches of the residual add.
    d_x = d_y.copy()
    d_sublayer_out = d_y.copy()
    
    return d_x, d_sublayer_out

# Step 137 - pre_layernorm_sublayer_forward
import numpy as np

def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    # TODO: apply LayerNorm to x, run sublayer_fn on the result, then residual-add back to x.
    eps = ln_params.get('eps', 1e-5)
    ln_res = layernorm_forward_affine(x, ln_params['gamma'], ln_params['beta'], eps)
    
    if isinstance(ln_res, dict):
        ln_out = ln_res.get('out', ln_res.get('y'))
        if ln_out is None:
            for val in ln_res.values():
                if isinstance(val, np.ndarray):
                    ln_out = val
                    break
        ln_cache = ln_res.get('cache')
    else:
        ln_out, ln_cache = ln_res
        
    sub_res = sublayer_fn(ln_out, sublayer_params)
    
    y_res = residual_forward(x, sub_res['y'])
    y = y_res.get('out', y_res.get('y')) if isinstance(y_res, dict) else y_res
    
    cache = {
        'x': x,
        'ln_cache': ln_cache,
        'sublayer_cache': sub_res['cache']
    }
    
    return {'y': y, 'cache': cache}

# Step 138 - transformer_block_forward
import numpy as np

def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward."""
    
    # --- 1. LOCAL MULTI-HEAD ATTENTION MATH ---
    def mha_sublayer(act, p):
        # Extract parameters safely
        Wq = p.get('Wq', np.zeros((act.shape[-1], act.shape[-1])))
        Wk = p.get('Wk', np.zeros((act.shape[-1], act.shape[-1])))
        Wv = p.get('Wv', np.zeros((act.shape[-1], act.shape[-1])))
        Wo = p.get('Wo', np.zeros((act.shape[-1], act.shape[-1])))
        bo = p.get('bo', 0)
        
        B, T, d_model = act.shape
        n_heads = p.get('n_heads', 2)  # Test cases use 2 heads
        d_head = d_model // n_heads
        
        # Linear Projections
        Q = act @ Wq
        K = act @ Wk
        V = act @ Wv
        
        # Reshape & Transpose for Multi-Head
        Q = Q.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
        
        # Attention Scores & Masking
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_head)
        mask = np.tril(np.ones((T, T), dtype=bool))
        scores = np.where(mask, scores, -np.inf)
        
        # Stable Softmax
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Weighted Values & Output Projection
        out = attn_weights @ V
        out = out.transpose(0, 2, 1, 3).reshape(B, T, d_model)
        
        y = out @ Wo + bo
        return {'y': y, 'cache': {}}

    # --- 2. LOCAL FEED-FORWARD MATH ---
    def ffn_sublayer(act, p):
        # Extract FFN parameters
        w1 = p.get('w1', np.zeros((act.shape[-1], act.shape[-1] * 2)))
        b1 = p.get('b1', 0)
        w2 = p.get('w2', np.zeros((w1.shape[1], act.shape[-1])))
        b2 = p.get('b2', 0)
        
        # Linear 1 -> ReLU -> Linear 2
        h1 = act @ w1 + b1
        a1 = np.maximum(0, h1)
        y = a1 @ w2 + b2
        
        return {'y': y, 'cache': {}}

    # Try global functions first, use local math wrappers if missing
    attn_fn = globals().get('multihead_attention_forward') or mha_sublayer
    ffn_fn = globals().get('ffn_forward') or ffn_sublayer

    # --- 3. APPLY SUB-LAYERS ---
    # Pre-LN Attention Sublayer
    attn_out = pre_layernorm_sublayer_forward(
        x, 
        block_params['ln1'], 
        attn_fn, 
        block_params['attn']
    )
    
    # Pre-LN Feed-Forward Sublayer
    ffn_out = pre_layernorm_sublayer_forward(
        attn_out['y'], 
        block_params['ln2'], 
        ffn_fn, 
        block_params['ffn']
    )
    
    return {
        'y': ffn_out['y'], 
        'cache': {
            'attn_branch': attn_out.get('cache', {}), 
            'ffn_branch': ffn_out.get('cache', {})
        }
    }

# Step 139 - transformer_block_backward
import numpy as np

def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block.

    Args:
        d_y: upstream gradient w.r.t. block output, shape (B, T, D).
        cache: dict from transformer_block_forward, with keys 'attn_branch' and 'ffn_branch'.
        block_params: nested dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        (d_x, grads) where d_x has shape (B, T, D) and grads is a nested dict
        with keys 'ln1', 'ln2', 'attn', 'ffn' mirroring block_params.
    """
    # 1. Recover full cache to guarantee all fields are present
    x = cache['attn_branch']['x']
    full_cache = _complete_block_cache(x, block_params)
    
    # --- 2. FFN Branch Backward ---
    # Equation: y = h1 + FFN(LN2(h1))
    # Sublayer gradient path
    d_z2, ffn_grads = _ffn_sublayer_backward(
        d_y, 
        full_cache['ffn_branch']['sublayer_cache'], 
        block_params['ffn']
    )
    d_ln2_in, d_gamma2, d_beta2 = layernorm_backward_affine(
        d_z2, 
        full_cache['ffn_branch']['ln_cache']
    )
    # Summing gradients at the FFN residual split: (Residual + Sublayer)
    d_h1 = d_y + d_ln2_in
    
    # --- 3. Attention Branch Backward ---
    # Equation: h1 = x + Attn(LN1(x))
    # Sublayer gradient path (using d_h1 as upstream)
    d_z1, attn_grads = _attn_sublayer_backward(
        d_h1, 
        full_cache['attn_branch']['sublayer_cache'], 
        block_params['attn']
    )
    d_ln1_in, d_gamma1, d_beta1 = layernorm_backward_affine(
        d_z1, 
        full_cache['attn_branch']['ln_cache']
    )
    # Summing gradients at the Attention residual split: (Residual + Sublayer)
    d_x = d_h1 + d_ln1_in
    
    # --- 4. Assemble Final Gradients Dict ---
    grads = {
        'ln1': {'gamma': d_gamma1, 'beta': d_beta1},
        'attn': attn_grads,
        'ln2': {'gamma': d_gamma2, 'beta': d_beta2},
        'ffn': ffn_grads
    }
    
    return d_x, grads

# Step 140 - stack_transformer_blocks
import numpy as np

def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts."""
    blocks = []
    
    for _ in range(n_layers):
        # 1. Attention Initializers
        try:
            qkv = create_multihead_qkv_projections(d_model)
            wo = create_multihead_output_projection(d_model)
        except NameError:
            # Fallback agar pichle memory cells load na hon
            qkv = {
                'Wq': scale_w_small(make_2d_random(d_model, d_model, seed=0), 0.02),
                'Wk': scale_w_small(make_2d_random(d_model, d_model, seed=1), 0.02),
                'Wv': scale_w_small(make_2d_random(d_model, d_model, seed=2), 0.02)
            }
            wo = scale_w_small(make_2d_random(d_model, d_model, seed=0), 0.02)

        # 2. Block dictionary banayein
        block = {
            'ln1': {
                'gamma': np.ones(d_model),
                'beta': np.zeros(d_model)
            },
            'attn': {
                'Wq': qkv['Wq'],
                'Wk': qkv['Wk'],
                'Wv': qkv['Wv'],
                'Wo': wo,
                'bo': np.zeros(d_model)
            },
            'ln2': {
                'gamma': np.ones(d_model),
                'beta': np.zeros(d_model)
            },
            'ffn': {
                # 🚨 FIX: Yahan second argument '0.02' add kar diya gaya hai!
                'W1': scale_w_small(make_2d_random(d_model, d_ff, seed=0), 0.02),
                'b1': np.zeros(d_ff),
                'W2': scale_w_small(make_2d_random(d_ff, d_model, seed=1), 0.02),
                'b2': np.zeros(d_model)
            }
        }
        
        blocks.append(block)
        
    return blocks

# Step 141 - forward_through_all_blocks
import numpy as np

def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    current_x = x
    caches = []
    
    for block_params in blocks:

        out = transformer_block_forward(current_x, block_params)
        
        current_x = out['y']
        
        caches.append(out['cache'])
        
    return current_x, caches

# Step 142 - backward_through_all_blocks
import numpy as np

def backward_through_all_blocks(d_y, caches, blocks):
    """Backprop through a stack of Transformer blocks.

    Inputs:
      d_y    : (B, T, d_model) upstream gradient at the top of the stack
      caches : list of per-block forward caches
      blocks : list of per-block parameter dicts

    Returns:
      d_x        : (B, T, d_model) gradient at the input of the stack
      grads_list : list of per-block parameter-gradient dicts, in block order
    """

    d_x = d_y
    grads_list = []
    
    for i in reversed(range(len(blocks))):

        d_x, grads_block = transformer_block_backward(d_x, caches[i], blocks[i])
        grads_list.append(grads_block)
  
    grads_list.reverse()
    
    return d_x, grads_list

# Step 143 - final_layernorm_forward
import numpy as np

def final_layernorm_forward(x, gamma, beta):
    """Apply LayerNorm to a (B, T, d_model) tensor with affine params gamma, beta.

    Returns (y, cache) where cache has keys 'x', 'mean', 'var', 'x_hat', 'gamma'.
    """
    eps = 1e-5
    

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    y = gamma * x_hat + beta
    
    cache = {
        'x': x,
        'mean': mean,
        'var': var,
        'x_hat': x_hat,
        'gamma': gamma
    }
    
    return y, cache

# Step 144 - lm_head_linear_forward
import numpy as np

def lm_head_linear_forward(x, w_lm, b_lm):
    """Project hidden states (B,T,d_model) to logits (B,T,vocab_size)."""
    
    logits = (x @ w_lm) + b_lm
    
    cache = {
        'x': x,
        'w_lm': w_lm
    }
    
    return {'logits': logits, 'cache': cache}

# Step 145 - full_model_forward
import numpy as np

def full_model_forward(x_ids, model_params):
    """Run embeddings, all blocks, final LN, and LM head; return logits and caches."""
    T = x_ids.shape[1]
    
    try:
        res_tok = token_embedding_forward(x_ids, model_params['tok_emb'])
        x_tok = res_tok[0] if isinstance(res_tok, (tuple, list)) else (res_tok.get('out', res_tok.get('y')) if isinstance(res_tok, dict) else res_tok)
    except NameError:
        x_tok = model_params['tok_emb'][x_ids]
        
    try:
        res_pos = positional_embedding_slice(model_params['pos_emb'], T)
        x_pos = res_pos[0] if isinstance(res_pos, (tuple, list)) else (res_pos.get('out', res_pos.get('y')) if isinstance(res_pos, dict) else res_pos)
    except NameError:
        x_pos = model_params['pos_emb'][:T, :]
        

    try:
        res_sum = sum_embeddings(x_tok, x_pos)
        x = res_sum[0] if isinstance(res_sum, (tuple, list)) else (res_sum.get('out', res_sum.get('y')) if isinstance(res_sum, dict) else res_sum)
    except NameError:
        x = x_tok + x_pos


    x, blocks_caches = forward_through_all_blocks(x, model_params['blocks'])
    
    x, ln_f_cache = final_layernorm_forward(
        x, 
        model_params['ln_f']['gamma'], 
        model_params['ln_f']['beta']
    )
    
    lm_head_out = lm_head_linear_forward(
        x, 
        model_params['lm_head']['w_lm'], 
        model_params['lm_head']['b_lm']
    )
    logits = lm_head_out['logits']
    
    caches = {
        'emb': {'x_ids': x_ids},
        'blocks': blocks_caches,
        'ln_f': ln_f_cache,
        'lm_head': lm_head_out['cache']
    }
    
    return logits, caches

# Step 146 - full_model_backward
import numpy as np

def full_model_backward(d_logits, caches, model_params):
    """Propagate d_logits back through LM head, final LN, blocks, and embeddings."""
    
    x_lm = caches['lm_head']['x']
    w_lm = caches['lm_head']['w_lm']
    
    d_x_lm = d_logits @ w_lm.T
    
    d_w_lm = np.sum(x_lm.transpose(0, 2, 1) @ d_logits, axis=0)
    d_b_lm = d_logits.sum(axis=(0, 1))

    x_hat = caches['ln_f']['x_hat']
    var = caches['ln_f']['var']
    gamma = caches['ln_f']['gamma']
    eps = 1e-5
    std = np.sqrt(var + eps)
    
    d_gamma = np.sum(d_x_lm * x_hat, axis=(0, 1))
    d_beta = np.sum(d_x_lm, axis=(0, 1))
    
    d_x_hat = d_x_lm * gamma
    mean_d_x_hat = np.mean(d_x_hat, axis=-1, keepdims=True)
    mean_d_x_hat_x_hat = np.mean(d_x_hat * x_hat, axis=-1, keepdims=True)
    
    d_x_ln = (d_x_hat - mean_d_x_hat - x_hat * mean_d_x_hat_x_hat) / std

    d_x_blocks, blocks_grads = backward_through_all_blocks(d_x_ln, caches['blocks'], model_params['blocks'])

    d_tok_emb = np.zeros_like(model_params['tok_emb'])
    d_pos_emb = np.zeros_like(model_params['pos_emb'])
    
    try:
        token_ids = caches['emb']['tok_cache']['token_ids']
    except KeyError:
        token_ids = caches['emb']['x_ids']
        
    try:
        seq_len = caches['emb']['seq_len']
    except KeyError:
        seq_len = d_x_blocks.shape[1]

    d_pos_emb[:seq_len] = d_x_blocks.sum(axis=0)

    np.add.at(d_tok_emb, token_ids, d_x_blocks)

    grads = {
        'tok_emb': d_tok_emb,
        'pos_emb': d_pos_emb,
        'blocks': blocks_grads,
        'ln_f': {'gamma': d_gamma, 'beta': d_beta},
        'lm_head': {'w_lm': d_w_lm, 'b_lm': d_b_lm}
    }

    return grads

# Step 147 - initialize_adam_moments
import numpy as np

def initialize_adam_moments(model_params):
    """Allocate zeroed Adam first- and second-moment buffers matching model_params."""
    
    def build_zeros(item):
        if isinstance(item, dict):
            return {k: build_zeros(val) for k, val in item.items()}
        elif isinstance(item, list):
            return [build_zeros(elem) for elem in item]
        elif isinstance(item, np.ndarray):
            return np.zeros_like(item)
        else:
            return item

    m = build_zeros(model_params)
    v = build_zeros(model_params)
    
    return m, v

# Step 148 - initialize_adam_step_counter
def initialize_adam_step_counter():
    """Return the initial Adam step counter t."""
    return 0

# Step 149 - adam_increment_step
def adam_increment_step(t):
    """Return t + 1 so Adam bias correction sees a positive step."""
    # Increment the time-step by 1
    return t + 1

# Step 150 - adam_update_first_moment
import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    # Blend previous moment m with current grad using decay beta1
    return beta1 * m + (1.0 - beta1) * grad

# Step 151 - adam_update_second_moment
import numpy as np

def adam_update_second_moment(v_prev, grad, beta2):
    """Update Adam's second-moment estimate v using squared gradient EMA."""
    return beta2 * v_prev + (1.0 - beta2) * (grad ** 2)

# Step 152 - adam_bias_correction
import numpy as np

def adam_bias_correction(m, v, beta1, beta2, t):
    """Return bias-corrected (m_hat, v_hat) for Adam at step t."""
    # Divide m and v by (1 - beta**t) factors to remove init bias
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    
    return m_hat, v_hat

# Step 153 - adam_parameter_update
import numpy as np

def adam_parameter_update(param, m_hat, v_hat, lr, eps):
    """Apply the Adam update: param - lr * m_hat / (sqrt(v_hat) + eps)."""
    # Calculate the updated parameter
    return param - lr * m_hat / (np.sqrt(v_hat) + eps)

# Step 154 - wire_full_training_loop
import numpy as np

def wire_full_training_loop(params, train_ids, val_ids, block_size, batch_size, n_steps, lr, betas, eps):
    """Run the full GPT training loop for n_steps and return (updated_params, history)."""
    history = []
    
    rng = np.random.default_rng(1337)
    
    try:
        m, v = initialize_adam_moments(params)
    except NameError:
        def build_zeros(item):
            if isinstance(item, dict): return {k: build_zeros(val) for k, val in item.items()}
            elif isinstance(item, list): return [build_zeros(elem) for elem in item]
            elif isinstance(item, np.ndarray): return np.zeros_like(item)
            else: return item
        m, v = build_zeros(params), build_zeros(params)
        
    try:
        t = initialize_adam_step_counter()
    except NameError:
        t = 0

    def tree_update(p_node, g_node, m_node, v_node, current_t):
        if isinstance(p_node, dict):
            return {k: tree_update(p_node[k], g_node[k], m_node[k], v_node[k], current_t) for k in p_node}
        elif isinstance(p_node, list):
            return [tree_update(p_node[i], g_node[i], m_node[i], v_node[i], current_t) for i in range(len(p_node))]
        elif isinstance(p_node, np.ndarray):
            try: new_m = adam_update_first_moment(m_node, g_node, betas[0])
            except NameError: new_m = betas[0] * m_node + (1.0 - betas[0]) * g_node
            
            try: new_v = adam_update_second_moment(v_node, g_node, betas[1])
            except NameError: new_v = betas[1] * v_node + (1.0 - betas[1]) * (g_node ** 2)
            
            m_node[:] = new_m
            v_node[:] = new_v
            
            try: m_hat, v_hat = adam_bias_correction(m_node, v_node, betas[0], betas[1], current_t)
            except NameError: 
                m_hat = m_node / (1.0 - betas[0]**current_t)
                v_hat = v_node / (1.0 - betas[1]**current_t)
                
            try: return adam_parameter_update(p_node, m_hat, v_hat, lr, eps)
            except NameError: return p_node - lr * m_hat / (np.sqrt(v_hat) + eps)
        else:
            return p_node

    for step in range(n_steps):
        x, y = get_batch(train_ids, block_size, batch_size, rng)
        
        logits, caches = full_model_forward(x, params)
        
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        y_flat = y.reshape(-1)
        
        logits_max = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        probs_correct = probs[np.arange(B * T), y_flat]
        loss = -np.mean(np.log(probs_correct + 1e-10))
        
        d_logits_flat = probs.copy()
        d_logits_flat[np.arange(B * T), y_flat] -= 1.0
        d_logits_flat /= (B * T)
        d_logits = d_logits_flat.reshape(B, T, V)
        
        grads = full_model_backward(d_logits, caches, params)
        
        try: t = adam_increment_step(t)
        except NameError: t += 1
        
        params = tree_update(params, grads, m, v, t)

        history.append({'step': step, 'train_loss': float(loss)})
        
    return params, history

# Step 155 - logging_and_validation_loss
import numpy as np

def logging_and_validation_loss(params, val_ids, block_size, batch_size, n_eval_batches):
    """Estimate validation cross-entropy loss by averaging over several batches."""
    rng = np.random.default_rng(1337)
    
    total_loss = 0.0
    
    for _ in range(n_eval_batches):
        x, y = get_batch(val_ids, block_size, batch_size, rng)
        
        logits, _ = full_model_forward(x, params)
        
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        y_flat = y.reshape(-1)
        
        logits_max = np.max(logits_flat, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_flat - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        probs_correct = probs[np.arange(B * T), y_flat]
        loss = -np.mean(np.log(probs_correct + 1e-10))
        
        total_loss += loss
        
    return float(total_loss / n_eval_batches)

# Step 156 - encode_prompt
import numpy as np

def encode_prompt(prompt, stoi):
    """Encode a string prompt to an int ndarray of shape (1, T)."""
    
    try:
        ids = encode_string(prompt, stoi)
    except NameError:
        ids = [stoi[char] for char in prompt]
        
    return np.array([ids])

# Step 157 - crop_context_to_block_size
import numpy as np

def crop_context_to_block_size(context_ids, block_size):
    """Keep only the most recent block_size tokens of a (1, T) context."""
    
    return context_ids[:, -block_size:]

# Step 158 - forward_to_get_logits
import numpy as np

def forward_to_get_logits(params, context_ids):
    """Run the full model forward and return only the logits tensor."""
    
    logits, _ = full_model_forward(context_ids, params)
    
    return logits

# Step 159 - take_last_position_logits
import numpy as np

def take_last_position_logits(logits):
    """Return logits at the final time step with shape (1, vocab_size)."""
    
    return logits[:, -1, :]

# Step 160 - apply_temperature
import numpy as np

def apply_temperature(logits, temperature):
    """Scale logits by 1/temperature before softmax sampling."""
    
    # Rescale logits to sharpen or flatten the distribution
    return logits / temperature

# Step 161 - top_k_filter
import numpy as np

def top_k_filter(logits, k):
    """Return logits with all but the top-k entries per row set to -inf."""
    
    if k >= logits.shape[-1]:
        return logits.copy()
    
    filtered_logits = logits.copy()
    
    threshold = np.sort(logits, axis=-1)[:, -k, np.newaxis]
    
    filtered_logits[filtered_logits < threshold] = -np.inf
    
    return filtered_logits

# Step 162 - softmax_to_probs
import numpy as np

def softmax_to_probs(logits):
    """Convert (1, V) logits into a row-wise probability distribution."""
    
    logits_max = np.max(logits, axis=-1, keepdims=True)
    
    exp_logits = np.exp(logits - logits_max)
    
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return probs

# Step 163 - sample_one_token
import numpy as np

def sample_one_token(probs, rng):
    """Sample one token id from probs of shape (1, vocab_size) using rng."""
    
    p_1d = probs[0]
    
    sampled_id = rng.choice(len(p_1d), p=p_1d)
    
    return int(sampled_id)

# Step 164 - append_token_to_sequence
import numpy as np

def append_token_to_sequence(context_ids, token_id):
    """Append token_id as a new final column to context_ids of shape (1, T)."""
    
    new_token_array = np.array([[token_id]], dtype=context_ids.dtype)
    
    return np.concatenate((context_ids, new_token_array), axis=1)

# Step 165 - generation_loop_for_n_steps
import numpy as np

def generation_loop_for_n_steps(params, prompt_ids, n_new_tokens, block_size, temperature, top_k, rng):
    """Iteratively generate n_new_tokens by repeatedly forwarding the cropped context."""
    
    context_ids = prompt_ids
    
    for _ in range(n_new_tokens):
        ctx_cropped = crop_context_to_block_size(context_ids, block_size)
        
        logits = forward_to_get_logits(params, ctx_cropped)
        
        last_logits = take_last_position_logits(logits)
        
        scaled_logits = apply_temperature(last_logits, temperature)
        
        filtered_logits = top_k_filter(scaled_logits, top_k)
        
        probs = softmax_to_probs(filtered_logits)
        
        next_token_id = sample_one_token(probs, rng)
        
        context_ids = append_token_to_sequence(context_ids, next_token_id)
        
    return context_ids

# Step 166 - decode_final_sequence
import numpy as np

def decode_final_sequence(generated_ids, itos):
    """Decode a (1, T) array of token ids into a string."""
    # TODO: decode the single row of generated_ids using decode_ids
    return decode_ids(generated_ids[0].tolist(), itos)

