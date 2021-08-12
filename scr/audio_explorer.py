# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa as lb
import librosa.display as lbdisp
from glob import glob
from json import dump

from scipy.ndimage.measurements import label

try:
    from logger_creator import CreateLogger
except:
    from scr.logger_creator import CreateLogger

logger = CreateLogger('AudioExplorer', handlers=1)
logger = logger.get_default_logger()


class AudioExplorer:
    def __init__(self, directory: str, audio_dir: str = r'/wav/*.wav', tts_file: str = r'/trsTrain.txt') -> None:
        try:
            self.tts_dict = {}
            self.df = None
            self.audio_files = []
            self.audio_freq = []
            self.main_dir = directory
            self.audio_files_dir_list = glob(self.main_dir + audio_dir)
            self.tts_file = self.main_dir + tts_file
            logger.info('Successfully Created AudioExplorer Class')
            self.load()
            logger.info('Successfully Loaded Audio and TTS files')

        except Exception as e:
            logger.exception("Failed Creating the AudioExplorer Class")

    def load(self):
        self.load_tts()
        self.load_audio()

    def load_tts(self) -> None:
        try:
            with open(self.tts_file, encoding='UTF-8') as tts_handle:
                lines = tts_handle.readlines()
                for line in lines:
                    text, file_name = line.split('</s>')
                    text = text.replace('<s>', '').strip()
                    file_name = file_name.strip()[1:-1]
                    self.tts_dict[file_name] = text
        except FileNotFoundError as e:
            logger.exception(
                f'File {self.tts_file} doesnt exist in the directory')
        except Exception as e:
            logger.exception('Failed to Load Transliteration File')

    def get_tts(self) -> dict:
        try:
            return self.tts_dict
        except Exception as e:
            logger.exception('Failed to return Transliteration')

    def export_tts(self, file_name: str) -> None:
        try:
            with open(file_name, "w") as export_file:
                dump(self.tts_dict, export_file, indent=4, sort_keys=True)

            logger.info(
                f'Successfully Exported Transliteration as JSON file to {file_name}.')

        except FileExistsError as e:
            logger.exception(
                f'Failed to create {file_name}, it already exists.')
        except Exception as e:
            logger.exception('Failed to Export Transliteration as JSON File.')

    def check_tts_exist(self, file_name: str) -> bool:
        if(file_name in self.tts_dict.keys()):
            return True
        else:
            return False

    def get_tts_value(self, file_name: str) -> str:
        if(self.check_tts_exist(file_name)):
            return self.tts_file[file_name]
        else:
            return 'Unknown'

    def load_audio(self) -> None:

        try:
            audio_name = []
            audio_mode = []
            audio_amplitude_min = []
            audio_amplitude_max = []
            audio_amplitude_mean = []
            audio_amplitude_median = []
            audio_frequency = []
            audio_duration = []
            has_TTS = []
            tts = []

            for audio_file in self.audio_files_dir_list:
                audio_data, audio_freq = lb.load(
                    audio_file, sr=None, mono=False)
                # Append Loaded Audio File
                self.audio_files.append(audio_data)
                # Append Loaded Audio File frequency(Sampling Rate)
                self.audio_freq.append(audio_freq)
                # Audio_Name
                name = audio_file.split('wav')[-2]
                name = name[1:-1].strip()
                audio_name.append(name)
                # Audio Mode (Mono, Stereo)
                audio_mode.append(
                    'Mono' if audio_data.shape == 1 else 'Stereo')
                # Time in seconds
                audio_duration.append(round(lb.get_duration(audio_data), 3))
                # Minimum Audio Amplitude
                audio_amplitude_min.append(round(min(audio_data), 3))
                # Maximum Audio Amplitude
                audio_amplitude_max.append(round(max(audio_data), 3))
                # Mean Audio Amplitude
                audio_amplitude_mean.append(round(np.mean(audio_data), 3))
                # Median Audio Amplitude
                audio_amplitude_median.append(round(np.median(audio_data), 3))
                # Audio Frequency
                audio_frequency.append(audio_freq)
                # TTS
                tts_status = self.check_tts_exist(name)
                has_TTS.append(tts_status)
                # Add Transliteration
                if(tts_status):
                    tts.append(self.tts_dict[name])
                else:
                    tts.append(None)

            self.df = pd.DataFrame()
            self.df['Name'] = audio_name
            self.df['Channel'] = audio_mode
            self.df['Duration(sec)'] = audio_duration
            self.df['Frequency(Hz)'] = audio_frequency
            self.df['MinAmplitude'] = audio_amplitude_min
            self.df['MaxAmplitude'] = audio_amplitude_max
            self.df['AmplitudeMean'] = audio_amplitude_mean
            self.df['AmplitudeMedian'] = audio_amplitude_median
            self.df['HasTTS'] = has_TTS
            self.df['TTS'] = tts

        except Exception as e:
            logger.exception('Failed to Load Audio Files')

    def get_audio_info(self) -> pd.DataFrame:
        try:
            return self.df.drop('TTS', axis=1)
        except Exception as e:
            logger.exception('Failed to return Audio Information')

    def get_audio_info_with_tts(self) -> pd.DataFrame:
        try:
            return self.df
        except Exception as e:
            logger.exception('Failed to return Audio Information')

    def get_audio_files(self) -> list:
        try:
            return self.audio_files, self.audio_freq
        except Exception as e:
            logger.exception('Failed to return Audio Files')

    def get_audio_file(self, index: int):
        try:
            return self.audio_files[index], self.audio_freq[index]
        except IndexError as e:
            logger.exception(
                f"Audio Files only exist between 0 - {len(self.audio_files) - 1}")
        except Exception as e:
            logger.exception('Failed to return Audio File')

    def get_audio_file_info(self, index: int):
        try:
            return self.df.iloc[index, :]
        except IndexError as e:
            logger.exception(
                f"Audio Files only exist between 0 - {len(self.df) - 1}")
        except Exception as e:
            logger.exception('Failed to return Audio File')

    def get_total_zero_crossing(self, index: int) -> int:
        zero_crossings = lb.zero_crossings(self.audio_files[index], pad=False)
        return sum(zero_crossings)

    def get_audio_visualization(self, index: int, figsize: tuple = (14, 5)):
        try:
            fig, ax = plt.subplots(figsize=figsize)
            lbdisp.waveplot(
                self.audio_files[index], sr=self.audio_freq[index], ax=ax)
            ax.set(title='Sound Visualization',
                   xlabel='Time(s)', ylabel='Amplitude')
            ax.label_outer()
            return plt
        except IndexError as e:
            logger.exception(
                f"Audio Files only exist between 0 - {len(self.audio_files) - 1}")
        except Exception as e:
            logger.exception('Failed to Visualize Audio File')

    def get_harmonic_percussive_visualization(self, index: int, figsize: tuple = (14, 5)):
        try:
            fig, ax = plt.subplots(figsize=figsize)
            y_harm, y_perc = lb.effects.hpss(self.audio_files[index])
            lbdisp.waveplot(
                y_harm, sr=self.audio_freq[index], alpha=0.25, ax=ax, label='Harmonic')
            lbdisp.waveplot(
                y_perc, sr=self.audio_freq[index], color='r', alpha=0.5, ax=ax, label='Percussive')
            ax.set(title='Harmonic + Percussive Visualization',
                   xlabel='Time(s)', ylabel='Amplitude')
            ax.legend()
            ax.label_outer()
            return plt
        except IndexError as e:
            logger.exception(
                f"Audio Files only exist between 0 - {len(self.audio_files) - 1}")
        except Exception as e:
            logger.exception(
                'Failed to Visualize Harmonics and Percussiveness of the Audio File')

    def get_pitch_visualization(self, index: int, figsize: tuple = (14, 5)):
        try:
            pitches, magnitudes = lb.piptrack(
                y=self.audio_files[index], sr=self.audio_freq[index])
            # plt.subplot(212)
            # plt.show()
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(pitches)
            ax.set(title='Pitch Visualization', ylabel='Pitch')
            return plt
        except IndexError as e:
            logger.exception(
                f"Audio Files only exist between 0 - {len(self.audio_files) - 1}")
        except Exception as e:
            logger.exception(
                'Failed to Visualize the Pitch of the Audio File')

    def get_spectogram_visualization(self, index: int, y_axis: str = 'hz', figsize: tuple = (14, 5)):
        # y-axis can be log
        try:
            X = lb.stft(self.audio_files[index])
            Xdb = lb.amplitude_to_db(abs(X))
            plt.figure(figsize=figsize)
            lbdisp.specshow(
                Xdb, sr=self.audio_freq[index], x_axis='time', y_axis=y_axis)
            plt.title = 'Mel spectrogram'
            plt.colorbar(format='%+2.0f dB')
            return plt

        except IndexError as e:
            logger.exception(
                f"Audio Files only exist between 0 - {len(self.audio_files) - 1}")
        except Exception as e:
            logger.exception('Failed to Visualize Spectogram of Audio File')

    def get_mfcc_visualization(self, index: int, figsize: tuple = (14, 5)):
        try:
            mfccs = lb.feature.mfcc(
                self.audio_files[index], sr=self.audio_freq[index])
            fig, ax = plt.subplots(figsize=figsize)
            lbdisp.specshow(
                mfccs, sr=self.audio_freq[index], x_axis='time', ax=ax)
            ax.set(title='MFCC Visualization',
                   xlabel='Time(s)', ylabel='MFCC')
            return plt

        except IndexError as e:
            logger.exception(
                f"Audio Files only exist between 0 - {len(self.audio_files) - 1}")
        except Exception as e:
            logger.exception('Failed to Visualize MFCC of Audio File')

    def get_chroma_visualization(self, index: int, hop_length: int = 512, figsize: tuple = (14, 5)):
        try:
            chromagram = lb.feature.chroma_stft(
                self.audio_files[index], sr=self.audio_freq[index], hop_length=hop_length)
            fig, ax = plt.subplots(figsize=figsize)
            lbdisp.specshow(chromagram, x_axis='time', y_axis='chroma',
                            hop_length=hop_length, cmap='coolwarm', ax=ax)
            ax.set(title='Chroma Visualization', xlabel='Time(s)')
            return plt

        except IndexError as e:
            logger.exception(
                f"Audio Files only exist between 0 - {len(self.audio_files) - 1}")
        except Exception as e:
            logger.exception('Failed to Visualize Chroma of Audio File')


if __name__ == "__main__":
    ae = AudioExplorer(directory='../data/train')
    print(ae.get_tts())
    print(ae.get_audio_info())
