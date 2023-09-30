import pathlib
import numpy as np
import pandas as pd
from warnings import warn
from stresspred.preprocess.preprocess_utils import resample_nonuniform


def identify_header(path, n=5, th=0.9):
    # https://stackoverflow.com/questions/40193388/how-to-check-if-a-csv-has-a-header-using-python
    df1 = pd.read_csv(path, header="infer", nrows=n)
    df2 = pd.read_csv(path, header=None, nrows=n)
    sim = (df1.dtypes.values == df2.dtypes.values).mean()
    return "infer" if sim < th else None


def get_frame_start_stop(index, frame_len=180, hop_len=None, start_index=0):
    if hop_len is None:
        hop_len = frame_len * 0.5
    in_start_stop = []
    out_start_stop = []
    i = 0
    in_stop = -1
    while in_stop <= np.max(index) + 1:
        if i == 0:
            in_start = start_index  # np.min(index)
            out_start = in_start
        else:
            in_start = in_start + hop_len
            out_start = in_start + hop_len / 2
        in_stop = in_start + frame_len
        out_stop = in_stop - hop_len / 2
        in_start_stop.append([in_start, in_stop])
        out_start_stop.append([out_start, out_stop])
        i += 1
    out_start_stop[-1] = [out_start_stop[-1][0], np.max(index) + 1]
    in_start_stop = np.array(in_start_stop)
    out_start_stop = np.array(out_start_stop)
    return in_start_stop, out_start_stop


def get_frame(sig, index, start_stop, frame_n):
    frame_indices = np.where(
        (index >= start_stop[frame_n - 1][0]) & (index < start_stop[frame_n - 1][1])
    )
    return sig[frame_indices]


def append_samples_to_wav(
    samples, wav_path="out.wav", sampling_rate=1000, rewrite=False
):
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            "Error in append_samples_to_wav(): the 'soundfile' module is required",
        )
    # if parent path does not exist, create it
    pathlib.Path(wav_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    if pathlib.Path(wav_path).is_file() and rewrite == False:
        with sf.SoundFile(wav_path, mode="r+") as wfile:
            wfile.seek(0, sf.SEEK_END)
            wfile.write(samples)
    else:
        sf.write(
            wav_path, samples, samplerate=sampling_rate, format="wav"
        )  # writes to the new file


def append_to_np_txt(
    a,
    txt_path="out.txt",
    fmt="%s",
    delimiter="\t",
    newline="\n",
    newline_on_append=True,
    rewrite=False,
):
    # if parent path does not exist, create it
    pathlib.Path(txt_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    if pathlib.Path(txt_path).is_file() and rewrite == False:
        f = open(txt_path, "a")
        np.savetxt(f, a, fmt=fmt, delimiter=delimiter, newline=newline)
        if newline_on_append:
            f.write(newline)
        f.close()
    else:
        np.savetxt(txt_path, a, fmt=fmt, delimiter=delimiter, newline=newline)


def write_sig_to_wav(
    sig,
    sig_time=None,
    wav_path="out.wav",
    old_sampling_rate=1000,
    new_sampling_rate=8000,
    frame_len=np.inf,
):

    sig = sig - np.nanmean(sig)
    sig[np.isnan(sig)] = 0
    sig = sig / np.max(np.abs(sig))
    in_start_stop, out_start_stop = get_frame_start_stop(sig_time, frame_len=frame_len)
    for frame_n in np.arange(1, len(in_start_stop) + 1):
        frame_sig = get_frame(sig, sig_time, in_start_stop, frame_n)
        frame_sig_time = get_frame(sig_time, sig_time, in_start_stop, frame_n)
        if old_sampling_rate == new_sampling_rate:
            frame_sig_r = frame_sig
            frame_sig_time_r = frame_sig_time
        else:
            frame_sig_r, frame_sig_time_r = resample_nonuniform(
                frame_sig, sig_time=frame_sig_time, new_sampling_rate=new_sampling_rate
            )
        out_frame_sig_r = get_frame(
            frame_sig_r, frame_sig_time_r, out_start_stop, frame_n
        )

        if frame_n == 1:
            append_samples_to_wav(
                out_frame_sig_r, wav_path, new_sampling_rate, rewrite=True
            )
        else:
            append_samples_to_wav(
                out_frame_sig_r, wav_path, new_sampling_rate, rewrite=False
            )
    if frame_len == np.inf:
        out_frame_sig_r_time = get_frame(
            frame_sig_time_r, frame_sig_time_r, out_start_stop, frame_n
        )
        sig_info = {}
        sig_info["sig"] = out_frame_sig_r
        sig_info["time"] = out_frame_sig_r_time
        sig_info["sampling_rate"] = new_sampling_rate
        return sig_info


def timestamps_to_audacity_txt(
    timestamp, txt_path="out.txt", label="timestamp", save=True, rewrite=False
):
    timestamp = np.array(timestamp, dtype=object)
    if len(timestamp.shape) == 1:
        start = timestamp
        end = timestamp
    else:
        start = timestamp[:, 0]
        end = timestamp[:, 1]
    if isinstance(label, str):
        labels = np.array([label] * len(timestamp))
    else:
        labels = label
    a = np.stack((start, end, labels), axis=1)
    if save:
        append_to_np_txt(
            a=a,
            txt_path=txt_path,
            fmt="%f\t" + "%f\t" + "%s",
            delimiter="\t",
            newline="\n",
            newline_on_append=False,
            rewrite=rewrite,
        )
    return a


def write_dict_to_json(d, json_path="out.json", fmt="%s", rewrite=False):
    try:
        from json_tricks import dump
    except ImportError:
        raise ImportError(
            "Error in write_dict_to_json(): the 'json_tricks' module is required",
        )
    
    # if parent path does not exist create it
    pathlib.Path(json_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    if not pathlib.Path(json_path).suffix == ".json":
        json_path = pathlib.Path(json_path + ".json") 
    if not pathlib.Path(json_path).is_file() or rewrite:
        with open(str(json_path), "w") as json_file:
            dump(d, json_file, allow_nan=True)
    else:
        warn("Warning: " + str(json_path) + " already exists.")