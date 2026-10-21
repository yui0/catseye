#!/usr/bin/env python3
"""Regenerate standalone Japanese Reader fine-tune notebook with embedded scripts."""
from pathlib import Path
import base64
import io
import nbformat as nb
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).parent
INCLUDE=['eg2_ja_vocab.py','eg2_mix_vocab.py','eg2_ja_table.py','eg2_ja_cache.py','eg2_ja_train.py',
         'eg2_ja_fleurs.py','convert_eg2_reader.py','convert_eg2_table.py','eg2_bundle.py']


def make():
    with io.BytesIO() as mem:
        with ZipFile(mem,'w',ZIP_DEFLATED,compresslevel=9) as z:
            for filename in INCLUDE:z.write(ROOT/filename,filename)
        bundle=base64.b64encode(mem.getvalue()).decode('ascii')
    cells=[
      nb.v4.new_markdown_cell('''# EmbeddingGemma 2 — 日本語 Reader 学習（Stage 35）

Google Colab で **音声 Encoder を固定**し、日本語辞書を作り直して **Reader を日本語音声／日本語転写ペアで追加学習**します。完成物は Cat's eye の C エンジン用 `reader.eg2r / word_table.eg2q / reader_vocab.txt` です。

**注意**：既存の英語 `reader.pt` と日本語辞書を差し替えるだけでは日本語を認識できません。このノートは学習を行います。短いデモデータで出来上がるモデルの認識精度は実用保証されません。**Colab GPU（可能なら L4/A100）** を推奨します。音声 Encoder は約3億パラメータです。

初期設定は動作確認用として日本語 FLEURS を少量読み込みます。本格的な認識にはより大量で多様な日本語音声・正解文章と学習時間が必要です。データセットのライセンス、Gemma の利用条件を確認してください。

既存 Stage 34 で作成した Encoder は再利用します。**このノートが作る ZIP には Encoder 重みは入りません**。最終 C 実行では Stage 34 の Encoder と組み合わせます。'''),
      nb.v4.new_code_cell('''# @title ① 設定
VOCAB_MODE = 'word'  # @param ['word','character']
MAX_VOCAB = 3500  # @param {type:'integer'}
TRAIN_LIMIT = 150  # @param {type:'integer'}
DEV_LIMIT = 30  # @param {type:'integer'}
EPOCHS = 3  # @param {type:'integer'}
BATCH_SIZE = 2  # @param {type:'integer'}
INITIALIZE_FROM_ENGLISH = True  # @param {type:'boolean'}
USE_FLEURS = True  # @param {type:'boolean'}
# USE_FLEURS=False 時は、別途 Colab に UTF-8 JSONL manifest + 音声を置いて以下を指定
TRAIN_MANIFEST = '/content/train.jsonl'  # @param {type:'string'}
DEV_MANIFEST = '/content/dev.jsonl'  # @param {type:'string'}
SAVE_TO_DRIVE = False  # @param {type:'boolean'}
BROWSER_DOWNLOAD = True  # @param {type:'boolean'}

from pathlib import Path
BASE=Path('/content/eg2_ja')
TOOLS=BASE/'tools'; MODEL=BASE/'source_encoder'; DATA=BASE/'dataset'
VOCAB=BASE/'reader_vocab.txt'; TABLE=BASE/'word_table.npy'
CACHE_TRAIN=BASE/'cache/train'; CACHE_DEV=BASE/'cache/dev'
TRAINED=BASE/'trained'; READY=BASE/'models-ja'; OUTPUT=BASE/'embeddinggemma2-reader-ja.zip'
assert VOCAB_MODE in ('word','character') and MAX_VOCAB>0
import torch
assert torch.cuda.is_available(), 'GPUランタイムを選択してください（ランタイム → ランタイムのタイプを変更）'
print('GPU:',torch.cuda.get_device_name(0))'''),
      nb.v4.new_code_cell('''# @title ② Python依存をインストール（実行時のみ）
import sys,subprocess
subprocess.run([sys.executable,'-m','pip','install','-q',
                'transformers>=5.19.0','sentence-transformers>=6.1.0',
                'huggingface_hub','datasets','soundfile','scipy',
                'sudachipy','sudachidict_core','torchcodec'],check=True)
for directory in (TOOLS,MODEL,DATA,CACHE_TRAIN,CACHE_DEV,TRAINED,READY):directory.mkdir(parents=True,exist_ok=True)
print('依存ライブラリ準備完了')'''),
      nb.v4.new_code_cell('''# @title ③ 日本語学習・変換ツールを展開（別のソースアップロード不要）
import base64,io,zipfile
BUNDLE = """'''+bundle+'''"""
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    expected={'''+','.join(repr(x) for x in INCLUDE)+'''}
    assert set(z.namelist())==expected
    z.extractall(TOOLS)
print('ツール:',len(expected))'''),
      nb.v4.new_markdown_cell('''## データとモデルの準備

`USE_FLEURS=True` は [google/fleurs の日本語音声](https://huggingface.co/datasets/google/fleurs) を少量取得します（CC BY 4.0）。自身のデータを使う場合は `USE_FLEURS=False` として、`{"id":"001", "audio":"/content/audio/001.wav", "text":"今日は晴れです"}` の JSONL を train と dev に用意してください。音声は WAV / FLAC 等、`soundfile` 対応形式です。'''),
      nb.v4.new_code_cell('''# @title ④ 日本語音声と転写を取得／読み込み
if USE_FLEURS:
    subprocess.run([sys.executable,'-u',str(TOOLS/'eg2_ja_fleurs.py'),
        '--out',str(DATA),'--train-limit',str(TRAIN_LIMIT),'--dev-limit',str(DEV_LIMIT)],check=True)
    TRAIN_MANIFEST=str(DATA/'train.jsonl');DEV_MANIFEST=str(DATA/'dev.jsonl')
assert Path(TRAIN_MANIFEST).is_file() and Path(DEV_MANIFEST).is_file()
print('学習用:',TRAIN_MANIFEST,'検証用:',DEV_MANIFEST)'''),
      nb.v4.new_markdown_cell('''必要なら Hugging Face の利用条件に同意し、認証を行ってください。下のセルでは `huggingface_hub.snapshot_download` を使い、**Encoder およびテキスト埋め込み用の公式重み**を取得します。Google のモデル利用規約・アクセス条件が適用されます。'''),
      nb.v4.new_code_cell('''# @title 任意：Hugging Face 認証（アクセスが拒否された場合だけ、このセルを手動で実行）
# from huggingface_hub import login
# login()
print('認証不要ならこのまま次へ進みます。')'''),
      nb.v4.new_code_cell('''# @title ⑤ EmbeddingGemma 2 公式モデルを取得
from huggingface_hub import snapshot_download,hf_hub_download
snapshot_download(repo_id='google/embeddinggemma-2',local_dir=str(MODEL),
    allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','merges.txt','vocab.*'])
print('encoder / embedding model:',MODEL)'''),
      nb.v4.new_code_cell('''# @title ⑥ 日本語辞書を生成（形態素＋文字フォールバック）
subprocess.run([sys.executable,str(TOOLS/'eg2_ja_vocab.py'),
    '--train',str(TRAIN_MANIFEST),'--mode',VOCAB_MODE,'--max-vocab',str(MAX_VOCAB),
    '--out',str(VOCAB)],check=True)
print('辞書行数:',len(VOCAB.read_text(encoding='utf8').splitlines()))'''),
      nb.v4.new_code_cell('''# @title ⑦ 日本語単語の768次元ベクトルを生成（実際のEmbeddingGemma 2を使用）
subprocess.run([sys.executable,'-u',str(TOOLS/'eg2_ja_table.py'),
    '--model-dir',str(MODEL),'--vocab',str(VOCAB),'--out',str(TABLE),
    '--device','cuda','--batch','32'],check=True)
print('word_table.npy 完成:',TABLE)'''),
      nb.v4.new_code_cell('''# @title ⑧ 日本語音声 → 512次元 Soft Tokens のキャッシュ（詳細ログ付き）
# 失敗時は最後のCalledProcessErrorだけでなく、内部Python tracebackを画面とログに保存。
# 既にできた .npy は再利用するので、失敗後の再実行で最初から計算し直しません。
import subprocess,sys

def run_cache_logged(split, manifest, out_dir):
    logfile=BASE/('eg2_cache_'+split+'.log')
    command=[sys.executable,'-u',str(TOOLS/'eg2_ja_cache.py'),
        '--manifest',str(manifest),'--model-dir',str(MODEL),
        '--out',str(out_dir),'--device','cuda','--dtype','auto']
    print('>>>',split,manifest,' | log:',logfile,flush=True)
    with logfile.open('w',encoding='utf-8') as log:
        with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',bufsize=1) as proc:
            for line in proc.stdout:
                print(line,end='',flush=True)
                log.write(line)
            returncode=proc.wait()
    if returncode:
        print('失敗時の詳細ログ:',logfile)
        print('Colabで確認: !tail -n 100 '+str(logfile))
        raise RuntimeError(f'{split} 音声キャッシュ失敗 (code={returncode})。上のtracebackを参照してください。')

for split,manifest,out_dir in [('train',TRAIN_MANIFEST,CACHE_TRAIN),('dev',DEV_MANIFEST,CACHE_DEV)]:
    run_cache_logged(split,manifest,out_dir)
print('音声特徴のキャッシュを生成しました')'''),
      nb.v4.new_code_cell('''# @title ⑨ 既存の英語 Reader v2 を初期値として取得（日本語用に追加学習する）
INIT=[]
if INITIALIZE_FROM_ENGLISH:
    pretrained=hf_hub_download(repo_id='mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader',filename='reader.pt')
    INIT=['--init-reader',str(pretrained)]
    print('英語Readerを初期値にします。日本語辞書のみの差し替えではありません。')
else:
    print('ランダム初期化から学習します。より多くの音声・計算が必要です。')'''),
      nb.v4.new_code_cell('''# @title ⑩ 日本語 Reader を CTC 追加学習
cmd=[sys.executable,'-u',str(TOOLS/'eg2_ja_train.py'),
    '--train',str(CACHE_TRAIN/'index.jsonl'),'--dev',str(CACHE_DEV/'index.jsonl'),
    '--vocab',str(VOCAB),'--table',str(TABLE),'--out',str(TRAINED),
    '--layers','6','--epochs',str(EPOCHS),'--batch',str(BATCH_SIZE),
    '--max-eval','30','--device','cuda',*INIT]
subprocess.run(cmd,check=True)
print('学習済み Reader:',TRAINED/'reader_ja.pt')'''),
      nb.v4.new_code_cell('''# @title ⑪ ネイティブC用 Reader／INT8辞書に変換して検査
import shutil,json,struct
subprocess.run([sys.executable,str(TOOLS/'convert_eg2_reader.py'),
    str(TRAINED/'reader_ja.pt'),str(READY/'reader.eg2r')],check=True)
subprocess.run([sys.executable,str(TOOLS/'convert_eg2_table.py'),
    str(TABLE),str(READY/'word_table.eg2q')],check=True)
shutil.copyfile(VOCAB,READY/'reader_vocab.txt')
with (READY/'word_table.eg2q').open('rb') as f:
    magic,version,nwords,width=struct.unpack('<4sIII',f.read(16))
assert magic==b'EG2Q' and version==1 and width==768
assert nwords==len(VOCAB.read_text(encoding='utf8').splitlines())
assert (READY/'word_table.eg2q').stat().st_size==16+nwords*768
shutil.copyfile(TRAINED/'training_info.json',READY/'training_info.json')
print('native Japanese Reader:',nwords,'tokens')'''),
      nb.v4.new_code_cell('''# @title ⑫ 日本語Readerと辞書を1つの .eg2m にまとめてZIP化
import zipfile
subprocess.run([sys.executable, str(TOOLS/'eg2_bundle.py'), '--ja-dir',str(READY),
    '--out',str(READY),'--force'],check=True)
with zipfile.ZipFile(OUTPUT,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
    z.write(READY/'ja.eg2m','models/ja.eg2m')
print('完成:',OUTPUT, 'size MiB:', round(OUTPUT.stat().st_size/2**20,2))
print('Stage38 の encoder.eg2m と組み合わせて利用します。')
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    dest=Path('/content/drive/MyDrive/EmbeddingGemma2')
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUTPUT,dest/OUTPUT.name)
    print('Driveに保存:',dest/OUTPUT.name)
if BROWSER_DOWNLOAD:
    from google.colab import files
    files.download(str(OUTPUT))'''),
      nb.v4.new_markdown_cell('''## Cat's eye で使用

英語ノートから作成した `models/encoder.eg2m` と `models/en.eg2m` に、日本語ノートの `models/ja.eg2m` を追加します。

```bash
gcc -O3 -std=gnu11 -DCATS_SPEECH_NATIVE -o cats cats.c -lm
./cats -a eg2-native-asr -m models -L ja -w speech_ja.wav
```

辞書と単語表の順番は変更しないでください。日本語 CER は追加学習したデータを使い検証してください。小規模学習は実用精度を保証しません。''')
    ]
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'colab':{'name':'EmbeddingGemma2_Japanese_Reader_Colab.ipynb','provenance':[]}})
    nb.validate(book)
    path=ROOT.parent/'colab'/'EmbeddingGemma2_Japanese_Reader_Colab.ipynb';nb.write(book,path,ensure_ascii=False)
    print('wrote',path,len(cells),'cells')

if __name__=='__main__':make()
