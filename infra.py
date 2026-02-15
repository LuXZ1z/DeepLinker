import datetime
import os
import logging
import base64
import hashlib
import random
from colorama import Fore
import platform
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager
import tqdm


system = platform.system()
if 'windows' in system.lower():
    BatchFoldDatabaseDir = r"../Linker_Structure_Database/BatchFoldResult"
elif 'linux' in system.lower():
    BatchFoldDatabaseDir = r"../Linker_Structure_Database/BatchFoldResult"
    
random.seed(os.urandom(4))
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['font.sans-serif'] = "Arial"
# matplotlib.rcParams['font.family'] = "Consolas"

TitleFont = matplotlib.font_manager.FontProperties(size=40)
SubTitle1Font = matplotlib.font_manager.FontProperties(size=30)
SubTitle2Font = matplotlib.font_manager.FontProperties(size=20)
SubTitle3Font = matplotlib.font_manager.FontProperties(size=10)
DefaultFontsize = 5
DefaultFont = matplotlib.font_manager.FontProperties(size=DefaultFontsize)
Zorder_Grid = 0
Zorder_Pic = 100
Zorder_Text = 200

Encoding = "utf-8"
TimeZone = 'Asia/Shanghai'
RootDataDir = "./linker_datas/linker_root_data/"
CollectedDataDir = "./linker_datas/linker_collected_data/"
LinkerLearnPath = "./linker_datas/linkerLearn/"

TimeFormat = "%Y%m%d-%H%M%S"
TqdmBarFormat = '%s{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.CYAN, Fore.GREEN)

Well96_Name2Index = {
    'A01': 1, 'B01': 2, 'C01': 3, 'D01': 4, 'E01': 5, 'F01': 6, 'G01': 7, 'H01': 8,
    'A02': 9, 'B02': 10, 'C02': 11, 'D02': 12, 'E02': 13, 'F02': 14, 'G02': 15, 'H02': 16,
    'A03': 17, 'B03': 18, 'C03': 19, 'D03': 20, 'E03': 21, 'F03': 22, 'G03': 23, 'H03': 24,
    'A04': 25, 'B04': 26, 'C04': 27, 'D04': 28, 'E04': 29, 'F04': 30, 'G04': 31, 'H04': 32,
    'A05': 33, 'B05': 34, 'C05': 35, 'D05': 36, 'E05': 37, 'F05': 38, 'G05': 39, 'H05': 40,
    'A06': 41, 'B06': 42, 'C06': 43, 'D06': 44, 'E06': 45, 'F06': 46, 'G06': 47, 'H06': 48,
    'A07': 49, 'B07': 50, 'C07': 51, 'D07': 52, 'E07': 53, 'F07': 54, 'G07': 55, 'H07': 56,
    'A08': 57, 'B08': 58, 'C08': 59, 'D08': 60, 'E08': 61, 'F08': 62, 'G08': 63, 'H08': 64,
    'A09': 65, 'B09': 66, 'C09': 67, 'D09': 68, 'E09': 69, 'F09': 70, 'G09': 71, 'H09': 72,
    'A10': 73, 'B10': 74, 'C10': 75, 'D10': 76, 'E10': 77, 'F10': 78, 'G10': 79, 'H10': 80,
    'A11': 81, 'B11': 82, 'C11': 83, 'D11': 84, 'E11': 85, 'F11': 86, 'G11': 87, 'H11': 88,
    'A12': 89, 'B12': 90, 'C12': 91, 'D12': 92, 'E12': 93, 'F12': 94, 'G12': 95, 'H12': 96,
}
Well96_Index2Name = {
    1: 'A01', 2: 'B01', 3: 'C01', 4: 'D01', 5: 'E01', 6: 'F01', 7: 'G01', 8: 'H01',
    9: 'A02', 10: 'B02', 11: 'C02', 12: 'D02', 13: 'E02', 14: 'F02', 15: 'G02', 16: 'H02',
    17: 'A03', 18: 'B03', 19: 'C03', 20: 'D03', 21: 'E03', 22: 'F03', 23: 'G03', 24: 'H03',
    25: 'A04', 26: 'B04', 27: 'C04', 28: 'D04', 29: 'E04', 30: 'F04', 31: 'G04', 32: 'H04',
    33: 'A05', 34: 'B05', 35: 'C05', 36: 'D05', 37: 'E05', 38: 'F05', 39: 'G05', 40: 'H05',
    41: 'A06', 42: 'B06', 43: 'C06', 44: 'D06', 45: 'E06', 46: 'F06', 47: 'G06', 48: 'H06',
    49: 'A07', 50: 'B07', 51: 'C07', 52: 'D07', 53: 'E07', 54: 'F07', 55: 'G07', 56: 'H07',
    57: 'A08', 58: 'B08', 59: 'C08', 60: 'D08', 61: 'E08', 62: 'F08', 63: 'G08', 64: 'H08',
    65: 'A09', 66: 'B09', 67: 'C09', 68: 'D09', 69: 'E09', 70: 'F09', 71: 'G09', 72: 'H09',
    73: 'A10', 74: 'B10', 75: 'C10', 76: 'D10', 77: 'E10', 78: 'F10', 79: 'G10', 80: 'H10',
    81: 'A11', 82: 'B11', 83: 'C11', 84: 'D11', 85: 'E11', 86: 'F11', 87: 'G11', 88: 'H11',
    89: 'A12', 90: 'B12', 91: 'C12', 92: 'D12', 93: 'E12', 94: 'F12', 95: 'G12', 96: 'H12'
}

# UpProteinSeq = (
#     "MDYKDHDGDYKDHDIDYKDDDDKQVQLVESGGGLMQAGGSLRLSCAVSGR"
#     "TFSTAAMGWFRQAPGKEREFVAAIRWSGGSAYYADSVKGRFTISRDKAKN"
#     "TVYLQMNSLKYEDTAVYYCARTENVRSLLSDYATWPYDYWGQGTQVTVSS"
#     "YPYDVPDYA"
# )
# DownProteinSeq = (
#     "RKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTG"
#     "KLPVPWPTLVTTLTYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTISFK"
#     "DDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVY"
#     "ITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYL"
#     "STQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
# # )
# assert len(UpProteinSeq) == 159
# assert len(DownProteinSeq) == 237

UpProteinSeq = (
   "MDEFEMIKRNTSEIISEEELREVLKKDEKSAYIGFEPSGKIHLGHYLQIKK"
   "MIDLQNAGFDIIIELADLAAYLNQKGELDEIRKIGDYNKKVFEAMGLKAKY"
   "VYGSEFQLDKDYTLNVFRLALKTTLKRARRSMELIAREDENPKVAEVIYPI"
   "MQVNAAHYLGVDVAVGGMEQRKIHMLARELLPKKVVCIHNPVLTGLDGEGK"
   "MSSSKGNFIAVDDSPKEIRAKIKKAYCPAGVVEGNPIMEIAKYFLEYPLTI"
   "KRPEKSGGDLTVDSYEELESLFKNKEPPLRALKNAVAEELIKILEPIRKRL"
   "YPYDVPDYA"
)
DownProteinSeq = (
    "RKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTG"
    "KLPVPWPTLVTTLTYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTISFK"
    "DDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVY"
    "ITADKQKNGIKANFKIRHNVEDGSVQLADHYQQNTPIGDGPVLLPDNHYL"
    "STQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYKGSAWSHPQFEKGG"
    "GSGGGSGGSAWSHPQFEK"
)

def get_time():
    return datetime.datetime.now().strftime(TimeFormat)


def gen_id(seq: str) -> str:
    encoding = 'utf-8'
    id_seq = base64.b32encode(hashlib.sha256(seq.encode(encoding)).digest()).lower()[:12].decode(encoding)
    return id_seq


def create_logger(
        logger_name: str,
        logger_filepath: str = None,
):
    if logger_filepath is None:
        logger_filepath: str = os.path.join(LinkerLearnPath, "%s_%s.log" % (logger_name, get_time()))

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(logger_filepath)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    # log message
    # logger.debug('debug message')
    # logger.info('info message')
    # logger.warning('warning message')
    # logger.error('error message')
    # logger.critical('critical message')
    logger.info("Logger file '%s' created" % os.path.abspath(logger_filepath))
    return logger


def ConfirmYes(y='Yes', Test=None):
    if not Test:
        return
    s = ''
    while not y == s:
        s = input("Input " + y + " to continue:\n")


