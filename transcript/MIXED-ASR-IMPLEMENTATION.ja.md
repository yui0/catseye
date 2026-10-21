# 日英混合文字起こしの実装メモ

## 実行方法

```sh
make
./cats -a eg2-native-asr -m models -L auto -w speech_mixed.wav
```

`models/encoder.eg2m` は既存の共通音声エンコーダ、`models/mix.eg2m` は**日英をひとつのReaderで学習したもの**です。後者はソースZIPに含まれません。

## 変更点

- `cats.c`: `-L auto`（別名 `mixed`, `en-ja`）が、`mix.eg2m` を解決します。存在しない場合は明確にエラーを返します。英語専用・日本語専用の動作は維持します。
- `tools/eg2_mix_vocab.py`: 日本語のUTF-8と英語の単語境界 `▁` を共通語彙にします。`こんにちは hello world` → `こんにちは▁hello▁world`。空白のない `OpenAIを使う` は `▁openaiを使う`。
- `tools/eg2_ja_table.py`: `▁word` の埋め込み生成では、先頭の `▁` を除いて語を埋め込みます。モデルのクラスIDの順序は変えません。
- `tools/eg2_ja_train.py --mixed`: 英日混合の正解テキストを共通語彙でCTC学習します。英語Reader初期化は引き続き可能。
- `tools/eg2_ja_fleurs.py --language ja_jp|en_us`: デモの英語・日本語データセットを準備します。
- `tools/eg2_bundle.py --mix-dir`: 学習済みReaderを単一 `mix.eg2m` にパッケージ化します。
- `colab/EmbeddingGemma2_English_Japanese_Mixed_Colab.ipynb`: モデル取得、英日データ作成、語彙、単語ベクトル、音声キャッシュ、Reader学習、C形式変換、ZIP生成までのColabノートブックです。

## 動作確認

- `make GPU=0 OPENMP=0` — ビルド成功。
- `make test-smoke GPU=0` — Cテスト4件通過。
- `python3 test/test_eg2_mixed.py` — 語彙の往復・ZIPパッケージ確認。
- `test/test_*.py` — Pythonテスト8件通過。
- フルCテスト一括実行はこの環境で実行時間の制限により終了せず。**実際の混合音声の認識精度は未検証**。

## 認識精度上の注意

学習データが日本語のみ・英語のみの発話を含んでいるだけでも、バイリンガル音声認識の学習はできますが、同じ発話内で言語が切り替わる場合の性能までは保証できません。日本語・英語に加えて **「これは OpenAI の model です」などの混合言語の実際の録音と正解転写** を学習・評価に追加してください。

`-L auto` は英語・日本語それぞれのReaderを自動切り替えするものではありません。エンコーダは1回、学習済み共通Readerと単語テーブルも1つで推論します。個別に作成した `ja.eg2m` と `en.eg2m` を置くだけでは同一発話内の混合認識になりません。
