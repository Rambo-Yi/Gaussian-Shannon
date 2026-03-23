import numpy as np
import torch
from PIL import Image
from pyldpc import make_ldpc, encode, decode, get_message


def pseudo_random_flip_sign_np(array, flip_prob=0.5, seed=42):
    """Pseudo-randomly flip signs in NumPy array (flip between 1 and -1, uniform distribution)

    Args:
        array: Input NumPy array (composed of 1 and -1)
        flip_prob: Flip probability (default 0.5)
        seed: Random seed (default 42)

    Returns:
        Flipped NumPy array
    """
    # Use local random generator (doesn't affect global np.random)
    rng = np.random.RandomState(seed)
    random_mask = rng.rand(*array.shape) < flip_prob  # Generate random mask
    flipped_array = array * (1 - 2 * random_mask)  # Flip signs
    return flipped_array.astype(array.dtype)  # Maintain original data type


def pseudo_random_recover_sign_np(flipped_array, flip_prob=0.5, seed=42):
    """Recover pseudo-randomly flipped signs in NumPy array (requires same random seed and probability)

    Args:
        flipped_array: Flipped array (composed of 1 and -1)
        flip_prob: Must be same flip probability as encoding (default 0.5)
        seed: Must be same random seed as encoding (default 42)

    Returns:
        Recovered original array (composed of 1 and -1)
    """
    # Use local random generator (doesn't affect global np.random)
    rng = np.random.RandomState(seed)
    random_mask = rng.rand(*flipped_array.shape) < flip_prob  # Generate same random mask

    # Recovery logic: if position was flipped, flip again (multiply by -1)
    recovered_array = flipped_array * (1 - 2 * random_mask)
    return recovered_array.astype(flipped_array.dtype)


def ldpc_encode(message, batch_size, redundancy, CR):
    """
    Args:
        message: Original information bits
        batch_size: Batch size
        redundancy: Redundancy copies, default 16
    Returns:
        final_tensor: shape [batch_size, n*redundancy], contains redundant encoding
        H: Parity check matrix
        G: Generator matrix
    """
    # 1. Define parameters
    # d_v < d_c
    # m = (n * d_v) // d_c  Number of parity bits
    # k = n - m  Number of information bits
    snr = float('inf')

    if CR == 0.25:
        n = 1024  # Codeword length
        d_v = 3  # Variable node average degree
        d_c = 4  # Check node average degree
        pad_width = 2

    # 2. Create LDPC parity check matrix H and generator matrix G
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

    # Pre-generate noise for all redundant encodings (shape [redundancy, batch_size, n])
    noise_all = np.random.randn(redundancy, batch_size, n)

    # Initialize result array
    final_arrays = []

    for j in range(redundancy):
        # Encoding (all samples use the same encoding for the j-th redundancy)
        pad_message = np.pad(message, (0, pad_width), 'constant')  # Padding
        codeword = encode(G, pad_message, snr=snr)  # Noiseless source codeword 0->1. 1->-1.

        # Pseudo-random processing (all samples use same processing for j-th redundancy, different seeds for different redundancies)
        s = pseudo_random_flip_sign_np(codeword, seed=42 + j)

        # Determine sign and add pre-generated noise
        redundant_codeword = s * np.abs(noise_all[j])  # shape [batch_size, n]

        if j == 0:
            final_arrays = redundant_codeword
        else:
            final_arrays = np.concatenate([final_arrays, redundant_codeword], axis=1)

    # Convert to tensor
    final_tensor = torch.from_numpy(final_arrays).float()

    return final_tensor, H, G


def ldpc_decode(message_batch, H, G, redundancy, table_decision, snr):
    """
    Args:
        message_batch: shape [B, N*redundancy], B is batch size, N is codeword length, each sample has 16 redundancies
        H: Parity check matrix
        G: Generator matrix
        redundancy: Number of redundancy copies, default 16

    Returns:
        m_batch: shape [B, K], K is information bits length
    """
    message_batch = -message_batch.cpu().numpy()  # Take negative (commonly used in LLR decoding)
    message_batch = np.ascontiguousarray(message_batch, dtype=np.float64)  # Ensure C-contiguous

    decision_cnt = 0
    maxiter = 100

    decoded_batch = []
    for message in message_batch:
        # Split each sample evenly into 16 parts
        split_messages = np.array_split(message, redundancy)
        decoded = None
        success = False

        for j, split_msg in enumerate(split_messages):
            # Pseudo-random sequence decoding
            split_code = pseudo_random_recover_sign_np(split_msg, seed=42 + j)

            # Attempt decoding (using Belief Propagation BP)
            d = decode(H, split_code, snr, maxiter=maxiter)
            if np.any((H @ d) % 2 != 0):
                continue  # Skip if parity check fails

            # Extract information bits from decoded codeword
            m = get_message(G, d)
            m = m[:-(len(m) - 256)]
            decoded = np.where(m == 1, 0, 1)

            success = True
            break  # Break out of loop as soon as one copy succeeds

        # If all copies fail, use the result from the last copy
        # If all copies fail, perform majority voting
        if not success:
            if not table_decision:
                split_msg = split_messages[-1]
                # Pseudo-random sequence decoding
                split_code = pseudo_random_recover_sign_np(split_msg, seed=42 + redundancy - 1)
                # LDPC decoding
                d = decode(H, split_code, snr, maxiter=maxiter)
                m = get_message(G, d)
                m = m[:-(len(m) - 256)]
                decoded = np.where(m == 1, 0, 1)
                print("Error correction failed: decoding result does not satisfy parity check equation")
            else:
                # Majority voting
                print("Majority voting")
                decision_cnt += 1
                # Collect all redundancy versions
                all_recovered = []
                for j in range(redundancy):
                    split_code = pseudo_random_recover_sign_np(split_messages[j], seed=42 + j)
                    all_recovered.append(split_code)

                # Transpose matrix so each column represents all redundancy versions of the same bit position
                all_recovered = np.array(all_recovered).T

                # Perform majority voting for each bit position
                majority_voted = []
                for bits in all_recovered:
                    # Count number of 1s and -1s
                    count_pos = np.sum(bits > 0)
                    count_neg = np.sum(bits < 0)
                    # Choose majority
                    majority_bit = 1 if count_pos > count_neg else -1
                    majority_voted.append(majority_bit)

                # Convert majority voting result to numpy array
                majority_voted = np.array(majority_voted)

                # Attempt LDPC decoding with majority voting result
                d = decode(H, majority_voted, snr, maxiter=maxiter)
                if np.any((H @ d) % 2 != 0):
                    print("Parity check still fails after majority voting")

                m = get_message(G, d)
                m = m[:-(len(m) - 256)]
                decoded = np.where(m == 1, 0, 1)

        decoded_batch.append(decoded)

    return np.stack(decoded_batch, axis=0), decision_cnt  # Return [B, K] result


def ldpc_decode_t(message_batch, H, G, redundancy, table_decision, snr, path, batch_size, index):
    """
    Args:
        message_batch: shape [B, N*redundancy], B is batch size, N is codeword length,
                       each sample has 16 sets of redundancy.
        H: Parity check matrix.
        G: Generator matrix.
        redundancy: Number of redundant copies, default is 16.
    Returns:
        m_batch: shape [B, K], K is information bit length.
    """
    message_batch = -message_batch.cpu().numpy()  # Negate (commonly used for LLR decoding)
    message_batch = np.ascontiguousarray(message_batch, dtype=np.float64)  # Ensure C-contiguous

    decision_cnt = 0
    maxiter = 100

    decoded_batch = []
    for i, message in enumerate(message_batch):
        extra_batch = []
        # Evenly split each sample into 16 parts
        split_messages = np.array_split(message, redundancy)
        decoded = None
        success = False

        for j, split_msg in enumerate(split_messages):
            # Pseudo-random sequence decoding
            split_code = pseudo_random_recover_sign_np(split_msg, seed=42 + j)

            if j < 4:
                extra_batch.append(np.where(split_code >= 0, 0, 1))

            # Attempt decoding (using Belief Propagation BP)
            d = decode(H, split_code, snr, maxiter=maxiter)
            if np.any((H @ d) % 2 != 0):
                continue  # Skip if parity check fails

            # Extract information bits from decoded codeword
            m = get_message(G, d)
            m = m[:-(len(m) - 256)]
            decoded = np.where(m == 1, 0, 1)

            success = True
            # break  # Break loop if at least one copy succeeds

        extra_batch = extra_batch + extra_batch
        extra_batch[-2] = np.ones(1024)

        # If all copies fail, use the result of the last copy or perform majority voting
        if not success:
            if not table_decision:
                split_msg = split_messages[-1]
                # Pseudo-random sequence decoding
                split_code = pseudo_random_recover_sign_np(split_msg, seed=42 + redundancy - 1)
                # LDPC decoding
                d = decode(H, split_code, snr, maxiter=maxiter)
                m = get_message(G, d)
                m = m[:-(len(m) - 256)]
                decoded = np.where(m == 1, 0, 1)
                print("Decoding failure: result does not satisfy parity check equations")
            else:
                # Majority voting
                print("Majority voting")
                decision_cnt += 1
                # Collect all redundant versions
                all_recovered = []
                for j in range(redundancy):
                    split_code = pseudo_random_recover_sign_np(split_messages[j], seed=42 + j)
                    all_recovered.append(split_code)

                # Transpose matrix so each row represents all redundant versions of the same bit position
                all_recovered = np.array(all_recovered).T

                # Perform majority voting for each bit position
                majority_voted = []
                for bits in all_recovered:
                    # Count positive and negative values
                    count_pos = np.sum(bits > 0)
                    count_neg = np.sum(bits < 0)
                    # Choose the majority
                    majority_bit = 1 if count_pos > count_neg else -1
                    majority_voted.append(majority_bit)

                # Convert majority voting results to numpy array
                majority_voted = np.array(majority_voted)

                # extra_batch += [np.where(majority_voted == 1, 0, 1)] * 4
                extra_batch += [np.where(majority_voted == 1, 0, 1)]
                extra_batch += [np.ones(1024, dtype=np.int32)] * 3

                # Try LDPC decoding using majority voting results
                d = decode(H, majority_voted, snr, maxiter=maxiter)
                if np.any((H @ d) % 2 != 0):
                    print("Parity check failed even after majority voting")

                m = get_message(G, d)
                m = m[:-(len(m) - 256)]
                decoded = np.where(m == 1, 0, 1)

                # extra_batch += [d] * 4
                extra_batch += [d]
                extra_batch += [np.ones(1024, dtype=np.int32)] * 3

                # =======================
                all_data = np.concatenate(extra_batch)
                result = all_data.reshape(4, 64, 64) * 255
                result = result.astype(np.uint8)

                # Save each image individually
                for j, img in enumerate(result):
                    Image.fromarray(img).save(path + f'{batch_size * index + i}_{j}.png')

        decoded_batch.append(decoded)

    return np.stack(decoded_batch, axis=0), decision_cnt  # Returns [B, K] result


def gauss_encode(message, batch_size, redundancy):
    """
    Args:
        message: Original information bits
        batch_size: Batch size
        redundancy: Number of redundant copies, default is 16
    Returns:
        final_tensor: shape [batch_size, n*redundancy], containing redundant encoding
        H: Parity check matrix
        G: Generator matrix
    """
    n = 256
    # Pre-generate noise for all redundant encodings (shape [redundancy, batch_size, n])
    noise_all = np.random.randn(redundancy, batch_size, n)

    # Initialize results array
    final_arrays = []

    message = np.where(message == 0, -1, 1)  # 0->-1, 1->1

    for j in range(redundancy):
        # Pseudo-random processing (all samples' j-th redundancy uses the same processing,
        # but different redundancies of the same sample need different seeds)
        s = pseudo_random_flip_sign_np(message, seed=42 + j)

        # Determine sign and add pre-generated noise
        redundant_codeword = s * np.abs(noise_all[j])  # shape [batch_size, n]

        if j == 0:
            final_arrays = redundant_codeword
        else:
            final_arrays = np.concatenate([final_arrays, redundant_codeword], axis=1)

    # Convert to tensor
    final_tensor = torch.from_numpy(final_arrays).float()

    return final_tensor


def gauss_decode(message_batch, redundancy):
    message_batch = message_batch.cpu().numpy()

    decoded_batch = []
    for message in message_batch:

        # Evenly split each sample into redundancy parts
        split_messages = np.array_split(message, redundancy)

        # Collect all redundant versions
        all_recovered = []
        for j in range(redundancy):
            split_code = pseudo_random_recover_sign_np(split_messages[j], seed=42 + j)
            all_recovered.append(split_code)

        # Transpose matrix so each row represents all redundant versions of the same bit position
        all_recovered = np.array(all_recovered).T

        # Perform majority voting for each bit position
        majority_voted = []
        for bits in all_recovered:
            # Count positive and negative values
            count_pos = np.sum(bits > 0)
            count_neg = np.sum(bits < 0)
            # Choose the majority
            majority_bit = 1 if count_pos > count_neg else 0
            majority_voted.append(majority_bit)

        # Convert majority voting results to numpy array
        majority_voted = np.array(majority_voted)
        decoded_batch.append(majority_voted)

    return np.stack(decoded_batch, axis=0)  # Returns [B, K] result


def ml_snr_estimation(received_signal):
    """
    Estimate the SNR for a standard Gaussian signal of dimension B×C×H×W
    using Maximum Likelihood Estimation (MLE).

    Parameters:
    received_signal : numpy.ndarray or torch.Tensor
        The received signal with shape (B, C, H, W).

    Returns:
    snr_db : float
        Estimated SNR in decibels (dB).
    noise_var : float
        Estimated noise variance σ².
    """
    # Move to CPU and convert to numpy for processing
    received_signal = received_signal.cpu().numpy()
    y_flattened = received_signal.reshape(-1)  # Flatten to 1D array

    # Maximum Likelihood Estimation: σ² = max(0, mean(y²) - 1)
    total_power = np.mean(y_flattened ** 2)
    noise_var = max(total_power - 1, 1e-10)
    snr_db = 10 * np.log10(1 / noise_var)

    return snr_db, noise_var


def watermarkToLatents(wm, latents):
    """
    Maps the pre-processed watermark noise (wm) into the Latent space dimensions.

    Args:
        wm: torch.Tensor, shape [batch_size, n_redundant], already in modulated noise form.
        latents: torch.Tensor, shape [batch_size, C, H, W], the target Latent space.

    Returns:
        z_T: The completed initial Gaussian noise.
    """
    device = latents.device
    batch_size, C, H, W = latents.shape
    target_num_elements = C * H * W
    current_num_elements = wm.shape[1]

    # 1. Determine and pad dimensions
    if current_num_elements < target_num_elements:
        # Calculate the number of elements needed for padding
        padding_size = target_num_elements - current_num_elements

        extra_noise = torch.randn(batch_size, padding_size, device=device)
        full_wm = torch.cat([wm, extra_noise], dim=1)
    else:
        full_wm = wm

    # 2. Reshape to [batch_size, C, H, W]
    z_T = full_wm.view(batch_size, C, H, W)

    return z_T


def latentsToWatermark(num_elements, latents):
    """
    Args:
        num_elements: int, equivalent to wm.shape[1], representing the noise dimension length
                      used during watermark embedding.
        latents: torch.Tensor, shape [batch_size, C, H, W], typically the initial noise
                 restored via DDIM Inversion.

    Returns:
        final_tensor: The extracted noise sequence, shape [batch_size, num_elements].
    """
    batch_size = latents.shape[0]

    # 1. Flatten the Latent into a 1D vector sequence [batch_size, P]
    # Note: Flattening order must strictly match the view() order used during embedding
    # (default is C, H, W flattening).
    flattened_latents = latents.reshape(batch_size, -1)

    # 2. Slice and extract the corresponding noise signals from the first num_elements dimensions.
    # In Gaussian Shannon coding, this portion of noise contains the modulated watermark symbols.
    final_tensor = flattened_latents[:, :num_elements]

    return final_tensor