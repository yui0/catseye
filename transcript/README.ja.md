# Cat's eye — 文字起こし

[プロジェクト全体](../README.ja.md)

### 音声認識・ONNX・LLM のための小さな C 推論エンジン。実行時に Python は不要。

[![License: MPL 2.0](https://img.shields.io/badge/license-MPL--2.0-5b5bd6.svg)](#ライセンス)
[![Language](https://img.shields.io/badge/language-C11-20232a.svg)](#ファイル構成)
[![Acceleration](https://img.shields.io/badge/accel-AVX2%20%7C%20AVX%20%7C%20OpenMP%20%7C%20GPU-20232a.svg)](#速度)

[English](README.md)

**Cat's eye は、音声ファイルをオフラインで文字にします。** 中心は
[EmbeddingGemma 2 Transcribe](https://eg2.haku.sh/) のネイティブ実装です。音声エンコーダ、リーダー、
単語検索、デコードまで、すべて小さな実行ファイル1つで動き、PyTorch も ONNX Runtime もサーバーも要りません。
Python が必要なのは、モデルファイルを最初に変換するときだけです。

同じコードに、小さな ONNX ランタイムと RNN-T 音声認識、GGUF を読む LLaMA 系のテキスト生成も入っています。

> [!NOTE]
> **現状。** 英語は実モデルで最後まで動作確認済みです（225 秒の音声が正しい英文になります）。
> 日本語・日英混合は [`colab/`](colab/) のノートブックで Reader を学習する必要があります。この経路は実装済みですが、
> 精度は測っていません。単語誤り率も未測定です（元の開発者は英語で約 5% と報告しています）。

## すぐ使う

```bash
make                       # ./cats を作る（OpenMP。EGL/GLESv2 のヘッダーがあれば GPU 対応も）
```

**1. モデルを用意する**（最初の1回。Python と `huggingface_hub`・`torch`・`numpy` が必要）

```bash
python3 tools/prepare_eg2_native.py --download --preset official-v2 --out models
```

Hugging Face から公開済みの重みを取得し、次の2ファイルを作ります。
`models/encoder.eg2m`（共通の音声エンコーダ）と `models/en.eg2m`（英語のリーダーと語彙）。
変換済みの ZIP がある場合は `make bundle-english MODEL_ZIP=embeddinggemma2-native-official-v2.zip`。

**2. 文字起こしする**

```bash
./cats -a eg2-native-asr -w speech.wav            # 英語（./models を使う）
./cats -a eg2-native-asr -L ja -w speech_ja.wav   # 日本語（models/ja.eg2m が必要）
./cats -a eg2-native-asr -L auto -w mixed.wav     # 日本語＋英語（models/mix.eg2m が必要）
./cats -a eg2-native-asr -m /path/to/models -w speech.wav
```

WAV は 8/16/24/32bit の PCM または float、モノラルでも多チャンネルでも、サンプルレート・長さを問いません。
結果は標準出力に出ます。

**3. 任意：モデルのサイズを半分にする**

```bash
make fp16-models           # models_fp16/ を作る（エンコーダ 1.22 GB → 0.62 GB）
./cats -a eg2-native-asr -m models_fp16 -w speech.wav
```

大きな行列を float16 で保存し、読み込み時に float32 に戻します。英語モデルでは重みの相対誤差は 2e-7 で、
文字起こしは変わりませんでした。

## オプション

| オプション | 意味 |
|---|---|
| `-a eg2-native-asr` | WAV → 文字の全工程。 |
| `-m DIR` | モデルのフォルダ（既定 `models`）。 |
| `-w FILE` | 入力 WAV。 |
| `-L en\|ja\|auto` | リーダー選択（既定 `en`、`auto` は共通 `mix.eg2m`）。 |
| `-j N` | スレッド数。既定は物理コア数（または `OMP_NUM_THREADS`）。 |
| `-G auto\|off\|N` | GPU による行列積。`auto`（既定）はハードウェア GPU があれば使う。`N` は EGL デバイス番号。 |
| `-o FILE` | フレームごとのスコアも書き出す（デバッグ用）。 |

環境変数：`CATS_SIMD=scalar\|avx\|avx2\|avx512\|auto`、`OMP_NUM_THREADS`、`CATS_GPU=off\|auto\|N`。
長い音声の窓の重なり：`CATS_EG2_OVERLAP=N`（トークン数、96〜300、既定 96。25 トークン = 1 秒）。
窓の長さは 30 秒（750 トークン）がモデルの上限で、これ以上は延ばせません。エンコーダは局所・因果アテンションで
約 130 トークン分の履歴しか見ないため、窓を長くしても効きません。代わりに重なりを広げると、継ぎ目の単語の
聞き取りが良くなる可能性があります（`N=192` で継ぎ目ごとにエンコーダ履歴 144 トークン、Reader の右文脈 48 トークン、
長尺で計算量 約 +17%）。効果は未測定なので、同じ WAV で `CATS_EG2_OVERLAP=192 ./cats ...` と比べてください。
`./cats -a cpu-info` で CPU の対応状況、`./cats help` で全モードの一覧が出ます。

## 速度

48 kHz・225 秒の音声、Xeon E5-2697 v2（12 コア、AVX あり・AVX2/FMA なし）、Radeon HD 7870 XT での実測：

| ビルド | 時間 | 実時間比 |
|---|---|---|
| 初版（1 スレッド、1 トークンずつ） | 38 分 47 秒 | 実時間の約 10 倍かかる |
| 現在、CPU のみ | 約 33 秒 | 0.15 |
| 現在、GPU あり | 約 30 秒 | 0.13 |

ピークメモリは約 1.6 GB です。元になった Mac アプリは Apple Silicon で 15 分の音声を約 3 秒と報告していますが、
このプログラムは最新の GPU 経路がなければそこまでは出ません。

速くなった理由：

- **まとめて行列積。** 30 秒窓の 750 トークンを、各 Conformer 段で一度に処理します。キャッシュブロック化したタイルで
  （AVX2/FMA・AVX・スカラーを実行時に選択）、層ごとの 100 MB の重みをトークンごとではなく1回だけ読みます。
- **スレッド。** トークン・出力ブロック・語彙検索を OpenMP で並列化。ハイパースレッドは浮動小数点ユニットを共有して
  待ちが増えるだけなので、既定では使いません。
- **半精度モデル**（上記）で読み込むバイト数が半分になります。
- **GPU（任意）。** `cats_gpu.h` が同じ行列積を OpenGL ES 3.1 の compute shader として、ウィンドウなしの
  EGL デバイスで実行します（GLFW 不要）。重みは層を使っている間 GPU メモリに置きます。ソフトウェアラスタライザは
  除外し、GPU のエラーは CPU カーネルに自動で戻るので、結果が GPU に左右されることはありません。
  この環境ではシェーダが約 500 GFLOPS、CPU カーネルが約 210 GFLOPS です。全体の速度は他の工程が頭打ちにします。
  行を CPU と GPU で分け合うと遅くなりました（`CATS_GPU_SHARE` で調整できます）。

ビルドスイッチ：`make OPENMP=0`（1 スレッド）、`make GPU=0`（GPU コードなし）、`make GPU=1`（必須にする）。
どの組み合わせでも結果は浮動小数点の丸め誤差の範囲で一致します（単語は同じ、スコア差は約 0.01 以内）。

## 中身

| ファイル | 内容 |
|---|---|
| `cats.c` | `cats` コマンド本体。 |
| `cats_speech.h` | WAV 読み込みとリサンプル、log-mel 特徴量、音声区間検出。 |
| `cats_rnnt.h` | EmbeddingGemma 2 の音声エンコーダ・リーダー・単語検索・CTC、RNN-T ビーム探索。 |
| `cats_onnx.h` | ONNX パーサ、CPU SIMD カーネル、小さなグラフ実行系。 |
| `../cats_gpu.h` | 任意の GPU 行列積（EGL + OpenGL ES 3.1 compute）。 |
| `cats_hayamimi.h` | 任意の Sherpa-ONNX 連携（`-a asr`、`make hayamimi` でビルド）。 |
| `../catseye.h`, `../catseye_llm.h` | 元の Cat's eye のニューラルネット中核と LLM ランタイム。 |

各ヘッダーは単独で include できます。

## そのほかのモード

```bash
# テキスト生成（GGUF。llama2.c のチェックポイントは -c model.bin -z tokenizer.bin）
./cats -m model.gguf -n 256 -p "Once upon a time"
./cats -a chat -m model.gguf

# ONNX の調査と小さなモデルの実行
./cats -a onnx-info  -m model.onnx
./cats -a onnx-check -m model.onnx          # 未対応の演算子があると 0 以外で終了
./cats -a onnx-run   -m small.onnx -p 1,2,3

# encoder / decoder / joiner の ONNX による RNN-T 音声認識（実験的）
./cats -a asr-native -m encoder.onnx -D decoder.onnx -J joiner.onnx -z tokens.txt -w speech.wav

# 80 bin の log-mel 特徴量
./cats -a fbank -w speech.wav -o features.cfb
```

EmbeddingGemma 2 の段階別モード（`eg2-features`、`eg2-subsample`、`eg2-layer`、`eg2-conformer`、
`eg2-project`、`eg2-reader`、`eg2-decode`、`eg2-quantize`、`eg2-asr`）は1工程ずつ実行します。
デバッグや、新しいモデルを参照出力と照合するためのものです。

## モデル

| 項目 | 内容 |
|---|---|
| `encoder.eg2m` | 共通：Conv2D サブサンプラ、Conformer 12 層、投影、mel フィルタ。1.22 GB（float16 で 0.62 GB）。 |
| `en.eg2m` | 英語リーダー（約 1950 万パラメータ）、20 万語の INT8 テーブル、語彙。234 MB。 |
| `ja.eg2m` | Colab で学習した日本語リーダー。エンコーダは英語と共通です。 |
| `mix.eg2m` | Colab で日英混合データにより学習した**単一**Reader／共通語彙／INT8単語テーブル。 |

`.eg2m` は圧縮なしの単純なコンテナ（`EG2PKG1`）で、cats が1層ずつその場で読みます。便利なコマンド：

```bash
python3 tools/eg2_bundle.py --verify models/encoder.eg2m      # 中身を一覧
python3 tools/eg2_bundle.py --from-zip old.zip --out models   # 古い ZIP を詰め直す
python3 tools/prepare_eg2_native.py --list-models              # ほかの入手先とプリセット
```

展開済みのファイルには、従来の `-R reader.eg2r -Q table.eg2q -z vocab.txt` も使えます。

**日本語。** Google Colab で `colab/EmbeddingGemma2_Japanese_Reader_Colab.ipynb` を開きます。エンコーダを固定し、
日本語の語彙を作り直してリーダーを学習し、`ja.eg2m` を出力します。英語のパッケージを Colab で作る場合は
`colab/EmbeddingGemma2_Native_Colab.ipynb` を使います。

## 英語・日本語を同じ音声から同時認識する（混合言語）

`-L ja` と `-L en` は互いに独立した語彙・Reader を使うので、
**日本語文中の英単語（コードスイッチング）には十分ではありません**。
`-L auto` は日英で共用する単一の Reader と共通 CTC 語彙を使います。
音声エンコーダはそのまま再利用するため、言語検出・モデル切り替え・2回のエンコードは不要です。

1. Colab の [`EmbeddingGemma2_English_Japanese_Mixed_Colab.ipynb`](colab/EmbeddingGemma2_English_Japanese_Mixed_Colab.ipynb) を開き、**英語・日本語・日英混在の音声と正解の文字起こし**で学習します。デモは英語・日本語 FLEURS を組み合わせますが、これだけでは発話内混合の精度は保証できません。
2. Colab が出力する `embeddinggemma2-reader-en-ja.zip` から `models/mix.eg2m` を取り出し、既存の `models/encoder.eg2m` と並べます。追加の `en.eg2m`/`ja.eg2m` はなくても `-L auto` を使えます。
3. `make && ./cats -a eg2-native-asr -m models -L auto -w mixed.wav` で認識します。

既存のスクリプトを使って学習する場合、`tools/eg2_mix_vocab.py` で語彙を作り、
`eg2_ja_table.py` で 768 次元埋め込みを生成して、
`eg2_ja_train.py --mixed` で共通Readerを学習します。
`tools/eg2_bundle.py --mix-dir models-mix --out models` で `mix.eg2m` をまとめられます。
混合語彙の `▁` は主に英語単語の開始、または明示的な単語間の空白を示し、C 側の CTC 復号で半角スペースに変換されます。
日本語の文字は隙間なく結合します。

**重要**: `mix.eg2m` の重みはこのソース ZIP には含まれていません。
ファイル未作成の場合は明示的なエラーを返します。日英を分けて学習した Reader を単に
合体しても、混合言語モデルにはなりません。日本語CER、英語WER、混合文CER／言語切替地点を
独立の検証データで測定してください。テスト用の疑似重みだけで認識品質は検証できません。

## ファイル構成

```text
cats.c                     コマンドライン本体
cats_*.h  catseye*.h       ライブラリのヘッダー（上の表）
Makefile                   ビルド・テスト・モデル補助（make help）
run_tests.sh               C と Python の全テスト
tools/                     Python：モデル変換、バンドル、Colab 生成、日本語学習
colab/                     生成済みノートブック（作り直し：python3 tools/make_eg2_colab.py）
test/                      回帰テスト
fixtures/                  テストが使う小さな合成モデル
bench/                     マイクロベンチ（行列積・SIMD・INT8）
```

## テスト

```bash
make test-smoke            # 重点4件、数秒
make test                  # 全 C・Python テスト（環境で所要時間が変化）
./cats -a cpu-info
cc -O3 -fopenmp -I. bench/bench_eg2.c -lm -o bench_eg2 && ./bench_eg2   # 行列積の速度
```

テストは合成モデルを使うので、各工程が基準の計算どおりに動くことを確かめるもので、認識精度を測るものではありません。
GPU のテストは GPU がなければスキップされます。

## 困ったとき

| 症状 | 対処 |
|---|---|
| `invalid weights, audio or inference failure` | `models/encoder.eg2m` と `models/<言語>.eg2m` があり `eg2_bundle.py --verify` が通るか、入力が本当に WAV かを確認。 |
| `missing models/encoder.eg2m` | `-m DIR` を渡すか、`models/` のあるフォルダで実行。 |
| とても遅い | `./cats -a cpu-info` を確認。`scalar` なら SIMD が使われていません（`CATS_SIMD` を確認）。ほかの重い処理が動いているとスレッドの待ちも増えます。 |
| `cats-gpu: ... falling back to the CPU` | ドライバがシェーダを拒否しました。結果は正しいままです。消すなら `-G off`。 |
| `GLES3/gl32.h` でビルド失敗 | EGL/GLESv2 の開発ファイルを入れるか `make GPU=0`。 |
| 日本語が文字化け・崩れる | `-L ja` と、自分のデータで学習した `ja.eg2m` を組み合わせて使う。 |

## ライセンス

[MPL-2.0](https://mozilla.org/MPL/2.0/)。モデルの重みはそれぞれのライセンスに従います
（EmbeddingGemma 2 の重みは Google の Gemma 利用条件です）。

© 2016–2026 Yuichiro Nakada

### INT8エンコーダ

FP32またはFP16のエンコーダを、256要素ごとのスケール付きINT8へ変換できます。NumPyが必要です。

```sh
python3 tools/eg2_bundle.py --to-int8 models_fp16/encoder.eg2m --out models_int8
cp models_fp16/en.eg2m models_int8/
./cats -a eg2-native-asr -m models_int8 -w speech.wav
```

実モデルではエンコーダが618,731,120バイトから321,459,824バイトになります。
EG2L v3として大きな行列だけを量子化し、正規化係数・深さ方向畳み込み・クリップ境界はFP32を維持します。
読み込み時にFP32へ展開するため、推論中の重みメモリ量は減りません。
量子化は非可逆です。元モデルを残し、用途に合う音声で認識精度を比較してください。

長い音声の推論では、エンコーダの計算が終わってからReaderと語彙テーブルを読み込みます。
8窓ごとのバッチ処理とfloat32の行列計算を維持し、同時に保持する重みを減らしています。
225.475秒の音声では最大RSSが465.47 MiBから356.09 MiBへ約24%減少し、
処理時間は78.36秒から76.83秒、認識結果は全文一致でした。各1回の計測値です。
詳細は[メモリー計測結果](bench/int8_validation/memory.md)を参照してください。
