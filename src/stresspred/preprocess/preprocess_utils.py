import scipy
import numpy as np
import neurokit2 as nk

from warnings import warn
from stresspred.misc import get_func_kwargs, argtop_k
from neurokit2.hrv.intervals_utils import _intervals_successive
from neurokit2.signal import signal_interpolate


def peak_time_to_rri(peak_time, min_rri=None, max_rri=None):
    peak_time = np.sort(peak_time)
    rri = np.diff(peak_time) * 1000
    rri_time = peak_time[1:]
    if min_rri is None:
        min_rri = 0
    if max_rri is None:
        max_rri = np.inf
    keep = np.where((rri >= min_rri) & (rri <= max_rri))
    return rri[keep], rri_time[keep]


def rri_to_peak_time(rri, rri_time):
    if len(rri_time) < 1:
        return rri_time
    rri_time = rri_time[np.isfinite(rri)]
    rri = rri[np.isfinite(rri)]
    non_successive_rri_ind = np.arange(1, len(rri_time))[
        np.invert(_intervals_successive(rri, rri_time, thresh_unequal=10))
    ]
    subtr_time_before_ind = np.concatenate((np.array([0]), non_successive_rri_ind))
    times_to_insert = (
        rri_time[subtr_time_before_ind] - rri[subtr_time_before_ind] / 1000
    )
    peak_time = np.sort(np.concatenate((rri_time, times_to_insert)))
    return peak_time


def samp_to_timestamp(samp, sampling_rate=1000, sig_time=None):
    less_than_zero = np.where(samp < 0)
    if np.any(less_than_zero):
        warn(
            "Warning: the sample index is less than 0. Changing the sample index to 0."
        )
        samp[less_than_zero] = 0
    if sig_time is None:
        # sig_time = [1/sampling_rate] this assumption was wrong for Audacity
        timestamp = samp / sampling_rate - 1 / sampling_rate
    else:
        bigger_than_last_index = np.where(samp >= len(sig_time))
        if np.any(bigger_than_last_index):
            warn(
                "Warning: the sample index is more than the last index. Changing the sample index to the last index."
            )
            samp[bigger_than_last_index] = len(sig_time) - 1
        timestamp = sig_time[samp]
    return timestamp


def timestamp_to_samp(
    timestamp, sampling_rate=1000, sig_time=None, check_greater_than_last=True
):
    timestamp = np.array(timestamp)
    if timestamp.size == 1:
        timestamp = np.array([timestamp])
    if sig_time is None:
        # [1/sampling_rate] this assumption was wrong for Audacity
        sig_time = [0]
        if check_greater_than_last:
            warn(
                "Warning: to check whether the sample is greater than the last sample index, sig_time must be given"
            )
            check_greater_than_last = False
        samp = np.array(
            (timestamp - sig_time[0] + 1 / sampling_rate) * sampling_rate
        ).astype(int)
    else:
        samp = np.array([np.argmin(np.abs(sig_time - t)) for t in timestamp]).astype(
            int
        )

    if check_greater_than_last:
        greater_than_len = np.where(samp > len(sig_time) - 1)
        if np.any(greater_than_len):
            warn(
                "Warning: the sample index is greater than the last sample index. Changing the sample index to the last sample index."
            )
            samp[greater_than_len] = len(sig_time) - 1
    less_than_zero = np.where(samp < 0)
    if np.any(less_than_zero):
        warn(
            "Warning: the sample index is less than 0. Changing the sample index to 0."
        )
        samp[less_than_zero] = 0
    return samp


def sig_time_to_sampling_rate(
    sig_time, method="median", check_uniform=True, decimals=12
):
    if check_uniform:
        if not check_uniform_sig_time(sig_time, decimals=decimals):
            warn("Warning: the difference between timepoints is not uniform")
    if method == "mode":
        sampling_rate = int(1 / scipy.stats.mode(np.diff(sig_time)))
    else:
        sampling_rate = int(1 / np.median(np.diff(sig_time)))
    return sampling_rate


def sampling_rate_to_sig_time(sig, sampling_rate=1000, start_time=0):
    sig_time = (np.arange(0, len(sig)) / sampling_rate) + start_time
    return sig_time


def resample_nonuniform(
    sig, sig_time, new_sampling_rate=1000, interpolate_method="linear", use_matlab=True
):
    if use_matlab:
        import matlab.engine

        eng = matlab.engine.start_matlab()
        eng.workspace["x"] = matlab.double(np.vstack(sig).astype(dtype="float64"))
        eng.workspace["tx"] = matlab.double(np.vstack(sig_time).astype(dtype="float64"))
        eng.workspace["fs"] = matlab.double(new_sampling_rate)
        y, ty = eng.eval("resample(x,tx,fs);", nargout=2)
        new_sig = np.hstack(np.asarray(y))
        new_sig_time = np.hstack(np.asarray(ty))
        eng.quit()
    else:
        sampling_rate_interpl = sig_time_to_sampling_rate(
            sig_time, method="median", check_uniform=False
        )
        sig_interpl, sig_time_interpl = interpolate_nonuniform(
            sig,
            sig_time,
            sampling_rate=sampling_rate_interpl,
            method=interpolate_method,
        )
        new_n_samples = int(
            len(sig_time_interpl) * (new_sampling_rate / sampling_rate_interpl)
        )
        new_sig, new_sig_time = scipy.signal.resample(
            sig_interpl, new_n_samples, t=sig_time_interpl
        )
    return new_sig, new_sig_time


def interpolate_nonuniform(sig, sig_time, sampling_rate=1000, method="quadratic"):
    start_sample_new = np.floor(sampling_rate * sig_time[0])
    end_sample_new = np.ceil(sampling_rate * sig_time[-1])
    new_sig_time = np.arange(start_sample_new, end_sample_new + 1) / sampling_rate
    new_sig = signal_interpolate(
        x_values=sig_time, y_values=sig, x_new=new_sig_time, method=method
    )
    return new_sig, new_sig_time


def find_local_hb_peaks(
    peak_time,
    sig,
    sig_time=None,
    sampling_rate=1000,
    check_height_outlier=False,
    k_sample_ratio=0.5,
    use_prominence=False,
    **kwargs
):
    if sig_time is None:
        sig_time = sampling_rate_to_sig_time(sig=sig, sampling_rate=sampling_rate)
    else:
        sampling_rate = sig_time_to_sampling_rate(sig_time=sig_time)
    new_peak_time = []
    if check_height_outlier:
        peak_height = sig[timestamp_to_samp(peak_time, sampling_rate, sig_time)]
    for peak in peak_time:

        hb_sig, hb_sig_time = get_local_hb_sig(
            peak,
            sig=sig,
            sig_time=sig_time,
            sampling_rate=sampling_rate,
            **get_func_kwargs(get_local_hb_sig, **kwargs)
        )

        if check_height_outlier:
            if k_sample_ratio == 0:
                k = 1
            else:
                k = int(k_sample_ratio * len(hb_sig))

            if use_prominence:
                local_peaks, _ = scipy.signal.find_peaks(hb_sig)
                local_prominence = scipy.signal.peak_prominences(hb_sig, local_peaks)[0]
                potential_peaks_index = local_peaks[argtop_k(local_prominence, k=k)]
            else:
                potential_peaks_index = argtop_k(hb_sig, k=k)
            peak_is_outlier = True
            i = 0
            current_peak_index = np.nan
            while peak_is_outlier and i < len(potential_peaks_index):
                current_peak_index = potential_peaks_index[i]
                current_peak_height = hb_sig[current_peak_index]
                peak_height_with_current = peak_height.copy()
                peak_height_with_current = np.insert(
                    peak_height_with_current, 0, current_peak_height
                )
                # having a fit and predict class like sklearn estimator
                # would probably make this faster
                peak_is_outlier = nk.find_outliers(peak_height_with_current)[0]
                i += 1
                # alternatively instead of iterating through can make
                # sure that there are no two candidate peaks that are
                # spaced apart far enough to be S1 and S2
            if np.isnan(current_peak_index) or peak_is_outlier:
                new_peak = peak
            else:
                new_peak = hb_sig_time[current_peak_index]
        else:
            if len(hb_sig) > 1:
                if use_prominence:
                    local_peaks, _ = scipy.signal.find_peaks(hb_sig)
                    prominences = scipy.signal.peak_prominences(hb_sig, local_peaks)[0]
                    new_peak = hb_sig_time[local_peaks[np.argmax(prominences)]]
                else:
                    new_peak = hb_sig_time[np.argmax(hb_sig)]
            else:
                new_peak = peak
        new_peak_time.append(new_peak)

    new_peak_time = np.array(new_peak_time)
    return new_peak_time


def get_local_hb_sig(
    peak,
    sig,
    sig_time=None,
    sampling_rate=1000,
    time_before_peak=0.2,
    time_after_peak=0.2,
):
    hb_sig_indices = np.where(
        (sig_time > peak - time_before_peak) & (sig_time < peak + time_after_peak)
    )
    hb_sig = sig[hb_sig_indices]
    hb_sig_time = sig_time[hb_sig_indices]
    return hb_sig, hb_sig_time


def get_sig_time_ref_first_samp(sig_time, sig=None, missing_value=np.nan):
    if sig_time[0] > 0:
        sig_time = sig_time - sig_time[0]
    elif sig_time[1] == 0:
        n_zeros = len(np.where(sig_time == 0)[0])
        sampling_rate = sig_time_to_sampling_rate(sig_time)
        for_replace_zeros = np.concatenate(
            (np.array([0]), ([1 / sampling_rate] * (n_zeros - 1)))
        )
        replace_zeros = np.cumsum(for_replace_zeros)
        sig_time = (
            sig_time - sig_time[n_zeros] + np.max(replace_zeros) + 1 / sampling_rate
        )
        sig_time[0:n_zeros] = replace_zeros
        if sig is not None:
            sig[0:n_zeros] = [missing_value] * n_zeros
    else:
        sig_time = sig_time
    if sig is not None:
        return sig_time, sig
    return sig_time


def check_uniform_sig_time(sig_time, decimals=6):
    return len(np.unique(np.round(np.diff(sig_time), decimals=decimals))) == 1


def drop_missing(sig, sig_time=None, missing_value=np.nan):
    if np.isnan(missing_value):
        not_missing = np.invert(np.isnan(sig))
    else:
        not_missing = np.where(sig == missing_value)
    sig = sig[not_missing]
    if sig_time is not None:
        sig_time = sig_time[not_missing]
        return sig, sig_time
    return sig


def roll_func(x, window, func, func_args={}):
    roll_x = np.array(
        [func(x[i : i + window], **func_args) for i in range(len(x) - window)]
    )
    return roll_x


def abs_max(x):
    return x[np.argmax(np.abs(x))]


def detect_invert_ecg(sig, sampling_rate=1000):
    filt_sig = nk.signal.signal_filter(
        sig, sampling_rate=sampling_rate, lowcut=3, highcut=45
    )
    med_max = np.median(roll_func(filt_sig, window=1 * sampling_rate, func=abs_max))
    return med_max < np.mean(sig)


def invert_sig(sig):
    return sig * -1 + 2 * np.nanmean(sig)


def cut_to_same_time(a, b):
    a = list(a)
    b = list(b)
    max_min_time = np.max([np.min(a[1]), np.min(b[1])])
    min_max_time = np.min([np.max(a[1]), np.max(b[1])])
    for i in range(len(a)):
        a[i] = a[i][(a[1] >= max_min_time) & (a[1] <= min_max_time)]
        b[i] = b[i][(b[1] >= max_min_time) & (b[1] <= min_max_time)]
    return a, b


def find_shift(a, b):
    if len(a) > len(b):
        a = a[: len(a) - (len(a) - len(b))]
    else:
        b = b[: len(b) - (len(b) - len(a))]
    af = scipy.fft.fft(a)
    bf = scipy.fft.fft(b)
    c = scipy.fft.ifft(af * np.conj(bf))

    shift_samples = np.argmax(abs(c))
    return shift_samples


def cut_to_shift(sig, sig_time, shift_samples, sampling_rate):
    sig_time = sig_time - shift_samples / sampling_rate
    sig = sig[shift_samples:]
    sig_time = sig_time[shift_samples:]
    return sig, sig_time


def interpl_intervals_preserve_nans(x_old, y_old, x_new):
    x_old = x_old[np.isfinite(y_old)]
    y_old = y_old[np.isfinite(y_old)]
    y_new_nan = np.ones(x_new.size).astype(bool)
    step = np.median(np.diff(x_new))
    for i in range(len(x_old)):
        if i != 0:
            if np.abs((x_old[i] - (y_old[i] / 1000)) - x_old[i - 1]) < step:
                y_new_nan[
                    (x_new >= x_old[i] - (y_old[i] / 1000)) & (x_new <= x_old[i])
                ] = False
        y_new_nan[np.argmin(np.abs(x_new - x_old[i]))] = False
    f = scipy.interpolate.interp1d(
        x_old, y_old, kind="linear", fill_value="extrapolate"
    )
    y_new = f(x_new)
    y_new[y_new_nan] = np.nan
    return y_new


def norm_corr(a, b, maxlags=0):
    # https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
    Nx = len(a)
    if Nx != len(b):
        raise ValueError("a and b must be equal length")

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 0:
        raise ValueError("maxlags must be None or strictly positive < %d" % Nx)

    # c = c[Nx - 1 - maxlags:Nx + maxlags]
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, "full")
    # c = np.correlate(x, y, mode=2)
    c = c[Nx - 1 - maxlags : Nx + maxlags]
    return c


def a_moving_average(y, N=5):
    # https://stackoverflow.com/questions/47484899/moving-average-produces-array-of-different-length
    y_padded = np.pad(y, (N // 2, N - 1 - N // 2), mode="edge")
    y_smooth = np.convolve(y_padded, np.ones((N,)) / N, mode="valid")
    return y_smooth


# just realized that rather than a moving average over the general rris
# it only makes sense to do it over successive intervals


def scale_and_clip_to_max_one(
    x,
    min_value=0,
    replace_min_value=0,
    max_value=np.inf,
    replace_max_value=None,
    div_by_given_max=True,
):
    if replace_max_value is None:
        replace_max_value = max_value
    x[x < min_value] = replace_min_value
    x[x > max_value] = replace_max_value
    if div_by_given_max:
        return x / max_value
    else:
        return x / np.nanmax(x)


def find_anomalies(peak_time, sig_info=None, check_successive=True):
    peak_time = np.array(peak_time)

    rri, rri_time = peak_time_to_rri(peak_time, min_rri=60000 / 200, max_rri=60000 / 20)
    interpolation_rate = 4
    inter_f_time = np.arange(
        rri_time[0] - rri[0] / 1000 - 60 / 40,
        rri_time[-1] + 60 / 40,
        1 / interpolation_rate,
    )
    rri_mid_time = rri_time - (rri / 1000) / 2
    inter_rri = interpl_intervals_preserve_nans(rri_mid_time, rri, inter_f_time)
    diff_rri = np.diff(rri)
    if check_successive:
        diff_rri = diff_rri[find_successive_intervals(rri, rri_time)]
        diff_rri_time = rri_time[1:][find_successive_intervals(rri, rri_time)]
        successive_rri = rri[1:][find_successive_intervals(rri, rri_time)]

    diff_rri_mid_time = diff_rri_time - successive_rri / 1000
    inter_diff_rri = interpl_intervals_preserve_nans(
        diff_rri_mid_time, diff_rri, inter_f_time
    )

    max_rri = 60000 / 40
    min_rri = 60000 / 200
    max_theoretically_possible_diff_rri = max_rri - min_rri
    acceptable_diff_rri = 600
    median_rri = np.nanmedian(inter_rri)
    n_intervals = 2
    N = round(((median_rri / 1000) * n_intervals) / (1 / interpolation_rate))

    f1 = a_moving_average(
        scale_and_clip_to_max_one(
            np.abs(nk.standardize(inter_rri, robust=True)), min_value=1, max_value=10
        ),
        N=N,
    )
    n_intervals = 3
    N = round(((median_rri / 1000) * n_intervals) / (1 / interpolation_rate))
    f2 = a_moving_average(
        scale_and_clip_to_max_one(
            np.abs(nk.standardize(inter_diff_rri, robust=True)),
            min_value=1,
            max_value=10,
        ),
        N=N,
    )

    f3 = a_moving_average(
        scale_and_clip_to_max_one(
            np.abs(inter_diff_rri),
            min_value=acceptable_diff_rri,
            max_value=max_theoretically_possible_diff_rri,
        ),
        N=N,
    )

    peak_time_from_rri = rri_to_peak_time(rri, rri_time)
    if sig_info is not None:
        peak_height = sig_info["sig"][
            timestamp_to_samp(peak_time, sig_time=sig_info["time"])
        ]
        inter_peak_height = nk.signal_interpolate(peak_time, peak_height, inter_f_time)
        f4 = scale_and_clip_to_max_one(
            np.abs(nk.standardize(inter_peak_height, robust=True)),
            max_value=10,
            min_value=2,
        )
    else:
        f4 = np.zeros(f1.shape)

    f_sum = a_moving_average(
        np.nansum([f1 * 0.25, f2 * 0.25, f3 * 0.25, f4 * 0.25], axis=0), 2
    )
    excluded_peak_time = np.array(
        [peak for peak in peak_time if peak_time not in peak_time_from_rri]
    )
    f_sum[timestamp_to_samp(excluded_peak_time, sig_time=inter_f_time)] = 1
    return f_sum[timestamp_to_samp(peak_time, sig_time=inter_f_time)]


def fixpeaks_by_height(
    peak_time,
    sig_info=None,
    clean_sig_info=None,
    sig_name="zephyr_ecg",
    time_boundaries=None,
):
    new_peak_time = []
    if time_boundaries is None:
        time_boundaries = {}
        time_boundaries["before_peak_clean"] = 0.1
        time_boundaries["after_peak_clean"] = 0.1
        if sig_name == "zephyr_ecg":
            time_boundaries["before_peak_raw"] = (0.005,)
            time_boundaries["after_peak_raw"] = 0.005
        else:
            time_boundaries["before_peak_raw"] = (0.001,)
            time_boundaries["after_peak_raw"] = 0.001
    for seg_peak_time in peak_time:
        seg_sig = sig_info["sig"]
        seg_sig_time = sig_info["time"]
        sampling_rate = sig_info["sampling_rate"]
        if clean_sig_info is None:
            seg_clean_sig = nk.signal_filter(
                seg_sig,
                sampling_rate=sampling_rate,
                lowcut=0.5,
                highcut=8,
                method="butterworth",
                order=2,
            )
            seg_clean_sig_time = seg_sig_time
        else:
            seg_clean_sig = clean_sig_info["sig"]
            seg_clean_sig_time = clean_sig_info["time"]
        new_seg_clean_peak_time = find_local_hb_peaks(
            peak_time=[seg_peak_time],
            sig=seg_clean_sig,
            sig_time=seg_clean_sig_time,
            time_before_peak=time_boundaries["before_peak_clean"],
            time_after_peak=time_boundaries["after_peak_clean"],
            # check_height_outlier=True, # just realized this doesn't make sense when there's only one peak
        )
        new_seg_peak_time = find_local_hb_peaks(
            peak_time=new_seg_clean_peak_time,
            sig=seg_sig,
            sig_time=seg_sig_time,
            time_before_peak=time_boundaries["before_peak_raw"],
            time_after_peak=time_boundaries["after_peak_raw"],
        )
        new_peak_time.append(new_seg_peak_time)
