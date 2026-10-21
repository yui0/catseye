# EmbeddingGemma 2: 英語Reader固定 + 日本語Adapter + 共通CTC

## 何を修正したか

- `-L adapter` モードを追加。512次元音声特徴をエンコードした後、**元の英語Readerを1回だけ**実行し、その768次元出力を日本語Adapterに渡します。
- `models/en.eg2m`（Readerの全層、blank重み、約20万語のEG2Qテーブル）と`encoder.eg2m`は変更しません。
- 日本語Adapterは `tanh(W1[soft(512),h_en(768)]+b1)` → 残差`h_ja=normalize(h_en+W2*hidden+b2)`。言語ゲートは `sigmoid(w_gate·hidden+b_gate)`。`rank=64`ならAdapter重みは約0.53MB（日本語辞書は別）。
- CTCでは英語20万語の候補、日本語トークン、元の英語blankを1本の時系列に統合します。推論時は既存の高速な全語彙Top1探索を利用し、実験的なgreedy CTCで復号します。
- 学習時は英語200kクラスのlogsumexpを正確に計算して特徴とともにキャッシュし、共通CTCの分母で再利用します。日本語Adapter・ゲートのみ更新し、元の英語Readerの全層と英語テーブルは完全に固定します。
- 学習コードは英語Readerの元 `.pt` と `en.eg2m` 内の `reader.eg2r` が**バイト同一かチェック**します。英語の別モデルを混ぜて認識精度を壊さないための安全策です。
- 日本語の `word_table.npy` と日本語専用 `ja_adapter_vocab.txt` を作り、`ja_adapter.eg2a` にまとめます。日本語単語ベクトルは既存英語テーブルへは足さず、別の小テーブルとして収録されます。

## インストールと利用

```bash
cd catseye/transcript
make
# 事前に既存 encoder.eg2m / en.eg2m を models/ に置く。
# Colabで学習した ja_adapter.eg2a を models/ に置く。
./cats -a eg2-native-asr -m models -L en -w english.wav
./cats -a eg2-native-asr -m models -L adapter -w mixed.wav
# 変な言語切り替えが多ければ margin を上げる (default 0.08)
./cats -a eg2-native-asr -m models -L adapter -H 0.15 -w mixed.wav
```

構成:

```text
models/
  encoder.eg2m          (旧モデル・不変)
  en.eg2m               (約20万語・旧モデル・不変)
  ja_adapter.eg2a       (新規：小型Adapter + JA EG2Q + JA vocab)
```

このアーカイブにはモデル重みは含みません。`en.eg2m` の約20万語の表を再生成する必要はありません。

## Colab

`colab/EmbeddingGemma2_Japanese_Adapter_FrozenEnglish_Colab.ipynb` を実行します。

旧Colabで⑧の `cache/train/index.jsonl` と `cache/dev/index.jsonl` まで作成済みであることが前提です。Colabは同じランタイム内なら⑧から再開できます。新しい環境では元音声キャッシュを復元してください。

NotebookはT4（FP32）/L4/A100/CPUを自動判定。日本語専用語彙の作成→EmbeddingGemma2による日本語埋め込み表の作成→英語Readerの固定特徴キャッシュ→追加学習→エクスポートの順に実行します。元の英語ReaderをCTCで再学習するセルは使いません。

実験的パラメーター:

- `RANK=64`（16〜128で比較可能）
- `EPOCHS=12`, `BATCH_SIZE=2`
- 英語WER許容悪化 `EN_WER_TOLERANCE=0.01`（1パーセントポイント）
- 学習後、`history.json` の英語・日本語・混合の誤り率を個別に確認。

## 精度を守るための条件と現在の限界

1. **固定保証**：元の英語モデル（`-L en`）は従来どおりです。Adapter推論では言語ゲートが日本語と誤認すると、英語精度は落ち得ます。
2. **選択保証**：検証データで元の英語のWERから1ポイントを超えて劣化したepochは「最良モデル」として採用しません。実世界での精度保証ではありません。
3. **CTC整合性**：学習は英語全語彙の正確な分母を用いる一方、推論は既存CのTop1 greedy CTCです。Beam Search/LM補正は将来の改善対象。
4. **混合学習**：自然な日英コードスイッチの音声が重要です。FLEURSを単に連結した疑似混合音声だけでは自然な発話の間合いを十分に学習できません。
5. **語彙外**：英語の単語に原英語20万語の語彙外が含まれる場合は学習データから除外します。日本語の未知文字も同様です。ログに件数が出ます。
6. **既存重みとの照合**：別のReaderに英語モデルを入れ替えるなら英語特徴キャッシュを最初から作り直す必要があります。
7. **実機テスト**：C/Python特徴演算の一致、保存/読み込み、少量の合成学習と出力は確認していますが、T4実機での実音声学習、日英混合CERや英語WERの実際の改善幅は未測定です。

### ローカルの軽量テスト

```bash
make test-smoke
make test-adapter
```

## 改善（関連プロジェクトを参考に）

Embed2ASR（Reader に Conv1d ブロック + BiGRU）、KolStream、sea-asr（教師モデルの擬似ラベルと KenLM）を参考に、次を追加しました。

- **時間方向の文脈（EG2A v2）**: ランク空間に depthwise の時間畳み込みを追加（`--kernel`、既定 7 ≒ 280 ms）。
  初期値は恒等（中心タップ=1）で、`--kernel 1` なら従来の v1 と同一です。C 側は v1 / v2 の両方を読めます
  （v1 の `ja_adapter.eg2a` はそのまま使えます）。**v2 を使うには `cats` の再ビルドが必要**です。
  Python と C の出力は 1e-7 以内で一致を確認済みです。
- **日本語単語テーブルの学習**（既定オン。`--no-train-ja-table` で固定、`--lr-table`）。
- **EMA**（`--ema-decay`）、**warmup + cosine**（`--warmup`）、**勾配蓄積**（`--accum`）。
- 評価サンプルを言語ごと 100 件に増加。

未実装の候補: 日本語文字 n-gram LM + prefix beam search（Embed2ASR / sea-asr）、
教師モデル（Qwen3-ASR・Whisper 等）による擬似ラベルでの学習データ拡張（sea-asr）、
エンコーダ上位層の微調整（KolStream。ただし英語の完全固定は失われます）。
実音声での精度は未測定です。

## 擬似ラベルによる学習データ拡張（`tools/eg2_pseudo_label.py`）

sea-asr と同様に、ラベルなし音声を教師 ASR で書き起こして学習データにします。FLEURS だけでは足りないデータ量を補います。

```bash
pip install faster-whisper
# 1) 教師 #1（Whisper）で書き起こし、30 秒以下のクリップに分割
python tools/eg2_pseudo_label.py label --audio-dir raw/ --out-dir pseudo --language ja
# 2) pseudo/clips/*.wav を別の ASR（Qwen3-ASR 等）で書き起こし、{"id","text"} の JSONL にする
# 3) 2 つの教師が一致するクリップだけ残す（CER <= 0.10）。学習で使えない語彙の行も除外
python tools/eg2_pseudo_label.py filter --manifest pseudo/clips.jsonl --hyp hyp2.jsonl \
    --out pseudo/train.jsonl --en-vocab en_vocab.txt --ja-vocab reader_vocab.txt
```

その後は通常どおり `eg2_ja_cache.py` → `eg2_ja_adapter.py prepare/train` に渡します
（train.jsonl を FLEURS の train と連結）。

注意:
- **dev は人手ラベル（FLEURS など）のまま**にしてください。擬似ラベルで dev を作ると、英語 WER ガードも日本語 CER も信頼できなくなります。
- 教師の誤りは生徒に引き継がれます。2 教師の一致で減らせますが、ゼロにはなりません。
- 数字（`2024年` など）は ASCII 英数字として英語語彙と照合され、語彙外なら除外されます。
- 動作確認は合成データでの結合・一致率・フィルタのロジックのみです。Whisper での実行は未確認です。
