#!/usr/bin/env python3
"""Regenerate a self-contained Colab notebook from the local native model converters."""
import base64
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
NEEDED = [
    'prepare_eg2_native.py', 'check_eg2_native.py',
    'convert_eg2_subsample.py', 'convert_eg2_conformer.py',
    'convert_eg2_projection.py', 'convert_eg2_reader.py',
    'convert_eg2_table.py', 'eg2_bundle.py',
]


def cells():
    md = lambda s: {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(True)}
    code = lambda s: {'cell_type': 'code', 'execution_count': None, 'metadata': {},
                      'outputs': [], 'source': s.splitlines(True)}
    with BytesIO() as memory:
        with ZipFile(memory, 'w', compression=ZIP_DEFLATED, compresslevel=9) as z:
            for p in NEEDED:
                z.write(ROOT / p, p)
        payload = base64.b64encode(memory.getvalue()).decode('ascii')
    return [
        md('''# EmbeddingGemma 2 Transcribe — ColabでネイティブCモデルを作成

**Stage 38 / Stage 33互換**。Hugging Face の公開済みの学習済みモデルから、Cat’s eye のネイティブC推論用 `encoder.eg2m / en.eg2m` を作ります。

1. 下の設定でプリセットを選ぶ（`official-v2` 推奨）。
2. **ランタイム → すべてのセルを実行**。
3. 完了後に `embeddinggemma2-native-<preset>.zip` をダウンロードするか Google Drive に保存。

GPUは**必須ではありません**。このノートは**学習を行わず、公開学習済み重みをダウンロードして変換**します。新規学習/ファインチューニング用ではありません。

| プリセット | 音声 Encoder | 文字起こし Reader | 内容 |
|---|---|---|---|
| `official-v2` | [Google EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | 独立再現 v2（6層） | 標準・推奨 |
| `official-v1` | Google | 独立再現 v1（4層） | 過去版比較 |
| `unsloth-v2` | [Unsloth版](https://huggingface.co/unsloth/embeddinggemma-2) | 独立再現 v2 | 代替 Encoder（出力一致未検証） |

学習済み Reader は [公開された独立再現版](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader)です。元の Mac アプリに付属する非公開重みではありません。**英語用**です。

⚠️ Encoder は約 **1.49GB**、C向けの Conformer FP32層だけで約 **1.21GB**。ダウンロードや中間出力を含め、空き容量 **6GB以上（推奨8GB以上）** を確保してください。Colabランタイムの上限・ネットワーク状態により完了しないことがあります。実学習済み重みを使った C 側の文字起こし精度は、まだ別途検証が必要です。
'''),
        code('''# @title ① モデルと保存方法の選択
PRESET = "official-v2"  # @param ["official-v2", "official-v1", "unsloth-v2"]
DOWNLOAD_ONLY = False   # @param {type:"boolean"}
FORCE_REBUILD = False   # @param {type:"boolean"}
SAVE_TO_DRIVE = False   # @param {type:"boolean"}
DRIVE_FOLDER = "MyDrive/EmbeddingGemma2_Native"  # @param {type:"string"}
AUTO_BROWSER_DOWNLOAD = False  # @param {type:"boolean"}

from pathlib import Path
BASE = Path('/content/eg2_native')
MODELS = BASE / ('models-' + PRESET)
ARCHIVE = BASE / ('embeddinggemma2-native-' + PRESET + '.zip')
assert PRESET in ('official-v2', 'official-v1', 'unsloth-v2')
if SAVE_TO_DRIVE and (not DRIVE_FOLDER.startswith('MyDrive/') or '..' in Path(DRIVE_FOLDER).parts):
    raise ValueError('DRIVE_FOLDER は MyDrive/ 配下を指定してください')
print('モデル:', PRESET, '| 出力:', MODELS)
'''),
        code('''# @title ② Colab環境とライブラリを準備
import os, sys, subprocess, shutil, importlib.util
BASE.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(BASE).free / 1024**3
print(f'Colabの空き容量: {free:.2f} GiB')
if free < 5 and not DOWNLOAD_ONLY:
    raise RuntimeError('ディスク空き容量が不足しています。ランタイムを増やすか、不要ファイルを削除してください（目安6GB以上）。')
if importlib.util.find_spec('huggingface_hub') is None:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=True)
if not DOWNLOAD_ONLY:
    packages = [p for p in ('numpy','torch') if importlib.util.find_spec(p) is None]
    if packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True)
print('使用Python:', sys.version.split()[0])
print('GPU不要（重み変換はCPUを利用）')
'''),
        code('''# @title ③ Cモデル変換ツールをノートブックから展開（手動アップロード不要）
import base64, io, zipfile
BUNDLE_ZIP_BASE64 = """'''+payload+'''"""
TOOLS = BASE / 'tools'
TOOLS.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_ZIP_BASE64))) as z:
    expected = {'prepare_eg2_native.py', 'check_eg2_native.py', 'convert_eg2_subsample.py',
                'convert_eg2_conformer.py', 'convert_eg2_projection.py',
                'convert_eg2_reader.py', 'convert_eg2_table.py', 'eg2_bundle.py'}
    if set(z.namelist()) != expected:
        raise RuntimeError('ノートブック内の変換ツールが破損しています')
    z.extractall(TOOLS)
print('変換ツール:', len(expected), 'ファイルを展開しました')
'''),
        code('''# @title ④ ダウンロード予定のソースを確認
import subprocess, sys
subprocess.run([sys.executable, str(TOOLS / 'prepare_eg2_native.py'), '--list-models'], check=True)
subprocess.run([sys.executable, str(TOOLS / 'prepare_eg2_native.py'),
                '--download', '--dry-run', '--preset', PRESET, '--out', str(MODELS)], check=True)
print('Reader+語彙表+melフィルタを先に取得し、その後大容量Encoderを取得します。')
'''),
        md('''## Hugging Face 認証（必要な場合のみ）

公開リポジトリにアクセスできない場合は、Hugging Face のアカウントでログインする必要があります。通常は次のセルを飛ばせます。パスワードやトークンをノートブックの文字列に埋め込まないでください。'''),
        code('''# @title 任意：Hugging Faceにログインする（必要な場合だけ手動実行）
# Colabでこのセルを手動実行すると認証UIが表示されます。
from huggingface_hub import login
login()
'''),
        code('''# @title ⑤ モデルをダウンロードして C 用ファイルを作る
# ダウンロードした元ファイルは MODELS/_source/ に保存されます。
command = [sys.executable, '-u', str(TOOLS / 'prepare_eg2_native.py'),
           '--download', '--preset', PRESET, '--out', str(MODELS)]
if DOWNLOAD_ONLY:
    command.append('--download-only')
if FORCE_REBUILD:
    command.append('--force')
print('実行:', ' '.join(command), flush=True)
subprocess.run(command, check=True)
print('ダウンロードのみ完了' if DOWNLOAD_ONLY else '変換と構造検証が完了')
'''),
        code('''# @title ⑥ 完成モデルを再検査して配布用 ZIP を作る
import json, zipfile, hashlib, datetime
if DOWNLOAD_ONLY:
    print('DOWNLOAD_ONLY=True：ソースのみダウンロードしました。変換とZIP生成はスキップします。')
else:
    subprocess.run([sys.executable, str(TOOLS / 'check_eg2_native.py'),
        '--model-dir', str(MODELS), '--reader', str(MODELS/'reader.eg2r'),
        '--table', str(MODELS/'word_table.eg2q'), '--vocab', str(MODELS/'reader_vocab.txt'),
        '--require-mel'], check=True)
    # Automatic bundling already ran in prepare_eg2_native.py.
    names = ['encoder.eg2m', 'en.eg2m']
    for name in names:
        assert (MODELS/name).is_file(), name
    manifest = {
      'format': 'EG2PKG1', 'preset': PRESET, 'reader_language': 'en',
      'note': 'Structural validation only; actual word-error rate not evaluated',
      'files': {n: (MODELS/n).stat().st_size for n in names},
    }
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr('models/model_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        for name in names:
            z.write(MODELS / name, f'models/{name}')
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        assert all(f'models/{n}' in z.namelist() for n in names)
    print(f'完成: {ARCHIVE} ({ARCHIVE.stat().st_size / 1024**3:.2f} GiB)')
    print('モデルは encoder.eg2m と en.eg2m の2ファイルです')
'''),
        code('''# @title ⑦ ZIPの受け取り（Google Drive またはブラウザ）
if DOWNLOAD_ONLY:
    print('DOWNLOAD_ONLY=Trueです。必要なら DOWNLOAD_ONLY=False にして⑤以降を実行してください。')
else:
    from IPython.display import display, FileLink
    display(FileLink(str(ARCHIVE)))
    if SAVE_TO_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        dest_folder = Path('/content/drive') / DRIVE_FOLDER
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest = dest_folder / ARCHIVE.name
        print(f'Google Driveへ保存: {dest}')
        shutil.copyfile(ARCHIVE, dest)
        assert dest.stat().st_size == ARCHIVE.stat().st_size
        print('Google Driveに保存しました:', dest)
    if AUTO_BROWSER_DOWNLOAD:
        # 大きいZIPのブラウザダウンロードは失敗する場合があります。
        from google.colab import files
        files.download(str(ARCHIVE))
    print('PCでは ZIP を展開して models/ を C 推論プログラムへ渡します。')
    print('例: ./cats -a eg2-native-asr -m models -w speech.wav')
'''),
        md('''## 検証範囲・よくある問題

- **Google Colabで作るのは変換済みのモデルファイル**です。Readerのトレーニングは行いません。
- `HF_TOKEN` が必要なリポジトリの場合、Hugging Face側で利用条件を確認したうえで認証セルを使用してください。
- 変換が途中で失敗した場合、`models/_source/` の取得済みファイルが残ります。再開する際、変換済みの一部が残っていれば `FORCE_REBUILD=True` で⑤からやり直せます（既存の出力を上書きします）。
- `DOWNLOAD_ONLY=True` はダウンロード検査用です。完成したネイティブ形式が必要なら `False` にします。
- 元ファイルを含めるとZIPが巨大になるので、完成 ZIP は **C で使うモデル形式だけ**を含めます。
- `model_manifest.json` に参照元とファイルサイズを記録します。認識精度の保証ではありません。
- ZIPは大きいため、ブラウザより Google Drive への保存（`SAVE_TO_DRIVE=True`）を推奨します。
- **出力をColab内で実行検証したわけではありません。** 公開元のファイル内容が変更された場合、変換が失敗する可能性があります。

各モデルのライセンス・利用条件は配布元に従ってください。''')
    ]


def main():
    notebook_cells = cells()
    for i, cell in enumerate(notebook_cells):
        cell['id'] = f'eg2-colab-{i:02d}'
    notebook={'cells': notebook_cells, 'metadata': {
        'colab': {'provenance': [], 'name': 'EmbeddingGemma2_Native_Colab.ipynb'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    }, 'nbformat': 4, 'nbformat_minor': 5}
    dest = ROOT.parent / 'colab' / 'EmbeddingGemma2_Native_Colab.ipynb'
    dest.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(f'{dest}: {dest.stat().st_size:,} bytes')

if __name__=='__main__':
    main()
