#!/usr/bin/env python3
"""Make a standalone Colab notebook that trains a *single* en/ja CTC Reader.

The demo uses separate FLEURS ja/en speech utterances. Real intra-utterance
code switching also requires mixed-language audio/transcript training examples.
"""
from pathlib import Path
import nbformat as nb
from make_eg2_ja_colab import make as make_japanese

HERE = Path(__file__).resolve().parents[1] / 'colab'

def make():
    make_japanese()  # Generate current self-contained script bundle first.
    original = HERE / 'EmbeddingGemma2_Japanese_Reader_Colab.ipynb'
    book = nb.read(original, as_version=4)
    cells = book.cells
    assert len(cells) == 17 and 'eg2_ja_train.py' in cells[3].source
    cells[0].source = '''# EmbeddingGemma 2 — 日本語＋英語を同時認識する Reader 学習

共通の音声 Encoder と**日英共通の CTC Reader + 1つの語彙テーブル**を作ります。
言語ごとにファイルやパラメーターを切り替えず、`./cats -a eg2-native-asr -L auto` で認識する方式です。

デモとして FLEURS の日本語 `ja_jp` と英語 `en_us` を混合して学習しますが、
**各発話が単一言語のFLEURSだけでは、1発話内の日英切り替え性能は保証されません。**
日本語と英語が実際に混在する音声＋正解転写も追加して学習してください。

例：`今日はOpenAIのモデルをtestします` / `I use 日本語 recognition today`。
文字変換では英語の単語境界 `▁` を保持し、日本語には自動的な空白を挿入しません。

モデル重みは同梱しません。実用精度（日本語CER／英語WER／混合CER）は学習後に評価が必要です。
GPU ランタイムを推奨します。Google / Hugging Face / FLEURS の利用条件を確認してください。'''
    cells[1].source = '''# @title ① 設定（英日混合）
VOCAB_MODE = 'word'  # @param ['word','character']
MAX_VOCAB = 6000  # @param {type:'integer'}
TRAIN_LIMIT_PER_LANGUAGE = 500  # @param {type:'integer'}
DEV_LIMIT_PER_LANGUAGE = 70  # @param {type:'integer'}
EPOCHS = 5  # @param {type:'integer'}
BATCH_SIZE = 2  # @param {type:'integer'}
INITIALIZE_FROM_ENGLISH = True  # @param {type:'boolean'}
USE_FLEURS = True  # @param {type:'boolean'}
# USE_FLEURS=False: UTF-8 JSONL {"id":...,"audio":...,"text":...}
# 日本語・英語・日英混合の音声を含める。30秒以下の発話に分割。
TRAIN_MANIFEST = '/content/mixed_train.jsonl'  # @param {type:'string'}
DEV_MANIFEST = '/content/mixed_dev.jsonl'  # @param {type:'string'}
SAVE_TO_DRIVE = False  # @param {type:'boolean'}
BROWSER_DOWNLOAD = True  # @param {type:'boolean'}

from pathlib import Path
BASE=Path('/content/eg2_mix')
TOOLS=BASE/'tools'; MODEL=BASE/'source_encoder'; DATA=BASE/'dataset'
VOCAB=BASE/'reader_vocab.txt'; TABLE=BASE/'word_table.npy'
CACHE_TRAIN=BASE/'cache/train'; CACHE_DEV=BASE/'cache/dev'
TRAINED=BASE/'trained'; READY=BASE/'models-mix'; OUTPUT=BASE/'embeddinggemma2-reader-en-ja.zip'
assert VOCAB_MODE in ('word','character') and MAX_VOCAB>0
import torch
assert torch.cuda.is_available(), 'GPUランタイムを選択してください'
print('GPU:',torch.cuda.get_device_name(0))'''
    cells[4].source = '''## データを準備

`USE_FLEURS=True` なら FLEURS 日本語・英語をそれぞれ取得し、一つの学習リストに結合します。
同じ発話内の日本語＋英語にも対応させるには `USE_FLEURS=False` として、自分で録音した混合音声＋転写データを
`TRAIN_MANIFEST` と `DEV_MANIFEST` に指定してください。
必ず話者の違う検証データも用意してください（音声の学習・検証への重複投入は禁止）。'''
    cells[5].source = '''# @title ④ 日本語・英語データをそれぞれ取得して統合
import json
if USE_FLEURS:
    for language in ('ja_jp','en_us'):
        subprocess.run([sys.executable,'-u',str(TOOLS/'eg2_ja_fleurs.py'),
            '--out',str(DATA/language),'--language',language,
            '--train-limit',str(TRAIN_LIMIT_PER_LANGUAGE),
            '--dev-limit',str(DEV_LIMIT_PER_LANGUAGE)],check=True)
    # IDs must stay unique: the audio cache uses IDs as file names.
    for split in ('train','dev'):
        target=DATA/(split+'.jsonl')
        with target.open('w',encoding='utf8') as out:
            for language in ('ja_jp','en_us'):
                for row in (DATA/language/(split+'.jsonl')).read_text(encoding='utf8').splitlines():
                    item=json.loads(row);item['id']=language+'_'+item['id']
                    out.write(json.dumps(item,ensure_ascii=False)+'\\n')
    TRAIN_MANIFEST=str(DATA/'train.jsonl'); DEV_MANIFEST=str(DATA/'dev.jsonl')
assert Path(TRAIN_MANIFEST).is_file() and Path(DEV_MANIFEST).is_file()
print('学習:',TRAIN_MANIFEST,'検証:',DEV_MANIFEST)'''
    cells[9].source = '''# @title ⑥ 英日共通語彙を生成（▁は英語の単語境界）
subprocess.run([sys.executable,str(TOOLS/'eg2_mix_vocab.py'),
    '--train',str(TRAIN_MANIFEST),'--mode',VOCAB_MODE,
    '--max-vocab',str(MAX_VOCAB),'--out',str(VOCAB)],check=True)
print('共通語彙:',len(VOCAB.read_text(encoding='utf8').splitlines()))'''
    cells[10].source = cells[10].source.replace('日本語単語','日英共通単語')
    cells[11].source = cells[11].source.replace('日本語音声','日英の音声')
    cells[12].source = cells[12].source.replace('（日本語用に追加学習する）','（日英共通Readerに追加学習する）')
    cells[13].source = cells[13].source.replace('日本語 Reader','日英共通 Reader').replace("'--layers','6'", "'--mixed','--layers','6'")
    cells[14].source = cells[14].source.replace('native Japanese Reader:', 'native bilingual Reader:')
    cells[15].source = '''# @title ⑫ 学習済み共通Reader + 日英語彙を mix.eg2m にまとめる
import zipfile, shutil
subprocess.run([sys.executable,str(TOOLS/'eg2_bundle.py'),
    '--mix-dir',str(READY),'--out',str(READY),'--force'],check=True)
with zipfile.ZipFile(OUTPUT,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
    z.write(READY/'mix.eg2m','models/mix.eg2m')
print('完成:',OUTPUT, 'MiB:',round(OUTPUT.stat().st_size/2**20,2))
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    dest=Path('/content/drive/MyDrive/EmbeddingGemma2');dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUTPUT,dest/OUTPUT.name)
if BROWSER_DOWNLOAD:
    from google.colab import files
    files.download(str(OUTPUT))'''
    cells[16].source = '''## C 側で日本語・英語を同時認識

英語版 Colab で作成した `models/encoder.eg2m` を使い、
このノートで作成した `models/mix.eg2m` を同じディレクトリに配置します。

```bash
make
./cats -a eg2-native-asr -m models -L auto -w mixed.wav
```

`-L auto` は自動で Reader を切り替えるのではなく、**日英共通 Reader 1つ**でCTC復号します。
`en.eg2m` + `ja.eg2m` の2ファイルを置いただけでは混合音声にはなりません。
FLEURSだけでなく混合言語の録音で訓練し、3種類（日・英・混合）の評価を行ってください。'''
    # Colab GPU auto-selection: T4/Volta must use FP32, BF16-capable GPUs use BF16.
    # EmbeddingGemma 2 does not support FP16 activations; see official documentation.
    cells[0].source += "\n\nT4 (16GB VRAM) を含む NVIDIA GPU に対応します。BF16 非対応なら FP32 を選びます。CPU も実行可能ですが低速です。"
    old = "assert torch.cuda.is_available(), 'GPUランタイムを選択してください'\nprint('GPU:',torch.cuda.get_device_name(0))"
    assert old in cells[1].source
    cells[1].source = cells[1].source.replace("BATCH_SIZE = 2  # @param {type:'integer'}", """BATCH_SIZE = 2  # @param {type:'integer'}
DEVICE_PREFERENCE = 'auto'  # @param ['auto','cuda','cpu']
WORD_TABLE_BATCH = 16  # @param {type:'integer'}
EMBEDDING_DTYPE = 'auto'  # @param ['auto','fp32','bf16']""")
    cells[1].source = cells[1].source.replace(old, """if DEVICE_PREFERENCE not in ('auto','cuda','cpu'):
    raise ValueError('DEVICE_PREFERENCE: auto / cuda / cpu を選択')
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') if DEVICE_PREFERENCE == 'auto' else DEVICE_PREFERENCE
if DEVICE == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA がありません。Colab のランタイム > ランタイムのタイプを変更 > T4 GPU 等を選択してください')
if WORD_TABLE_BATCH < 1:
    raise ValueError('WORD_TABLE_BATCH は1以上にしてください')
if DEVICE == 'cuda':
    props = torch.cuda.get_device_properties(0)
    bf16 = torch.cuda.is_bf16_supported()
    print(f'GPU: {props.name} / {props.total_memory / 2**30:.1f} GiB / compute {props.major}.{props.minor} / BF16={bf16}')
    if EMBEDDING_DTYPE == 'bf16' and not bf16:
        raise ValueError('このGPUは BF16 非対応です。EMBEDDING_DTYPE=auto または fp32 を選択してください')
    print('EmbeddingGemma 2 演算精度:', 'BF16' if (EMBEDDING_DTYPE == 'bf16' or EMBEDDING_DTYPE == 'auto' and bf16) else 'FP32')
else:
    if EMBEDDING_DTYPE == 'bf16':
        raise ValueError('CPU 実行は EMBEDDING_DTYPE=auto または fp32 を選択してください')
    print('CPU モード: FP32 (遅くなります)')
print('選択デバイス:', DEVICE)""")
    assert "'--device','cuda','--batch','32'" in cells[10].source
    cells[10].source = cells[10].source.replace("'--device','cuda','--batch','32'", "'--device',DEVICE,'--dtype',EMBEDDING_DTYPE,'--batch',str(WORD_TABLE_BATCH)")
    assert "'--device','cuda','--dtype','auto'" in cells[11].source
    cells[11].source = cells[11].source.replace("'--device','cuda','--dtype','auto'", "'--device',DEVICE,'--dtype',EMBEDDING_DTYPE")
    assert "'--device','cuda',*INIT" in cells[13].source
    cells[13].source = cells[13].source.replace("'--device','cuda',*INIT", "'--device',DEVICE,*INIT")
    cells[16].source += "\n\nColab のランタイムで T4 GPU を選ぶと CUDA で動作します。初期値 `DEVICE_PREFERENCE='auto'` と `EMBEDDING_DTYPE='auto'` は T4 では FP32 に、BF16対応GPUでは BF16 に切り替わります。"
    book.metadata['colab']['name']='EmbeddingGemma2_English_Japanese_Mixed_Colab.ipynb'
    # Check that Python cells all parse; code strings need not be executable in local environment.
    import ast
    for n,cell in enumerate(cells):
        if cell.cell_type=='code':
            ast.parse(cell.source,filename=f'Colab cell {n}')
    nb.validate(book)
    output=HERE/'EmbeddingGemma2_English_Japanese_Mixed_Colab.ipynb'
    nb.write(book,output,ensure_ascii=False)
    print('wrote',output,len(cells),'cells')

if __name__=='__main__':make()
