import os
import random
from PIL import Image, ImageDraw, ImageFont
import torch
from library.train_util import ImageInfo, MinimalDataset


class JoyoKanjiDataset(MinimalDataset):
    # logsディレクトリに突っ込んである想定 / the file with letters is assumed to be in the logs directory
    LETTERS_FILE = "./logs/letters.txt"

    # フォントファイル、Google Fontsからダウンロードしてきたものを使う / use font files, downloaded from Google Fonts
    # ASLまたはOpen Font Licenseで配布されているものを使う / use those distributed under ASL or Open Font License
    FONT_FILES = [
        r"../dataset/HachiMaruPop-Regular.ttf",
        r"../dataset/KleeOne-SemiBold.ttf",
        r"../dataset/KosugiMaru-Regular.ttf",
        r"../dataset/NotoSansJP-Bold.otf",
        r"../dataset/NotoSansJP-Regular.otf",
        r"../dataset/NotoSansJP-Light.otf",
        r"../dataset/NotoSerifJP-Bold.otf",
        r"../dataset/NotoSerifJP-Regular.otf",
        r"../dataset/NotoSerifJP-Light.otf",
        r"../dataset/YujiBoku-Regular.ttf",
        r"../dataset/YujiMai-Regular.ttf",
        r"../dataset/YujiSyuku-Regular.ttf",
        r"../dataset/ZenKurenaido-Regular.ttf",
    ]

    # 学習時のプロンプトに使うフォント名 / font names used in training prompts
    FONT_NAMES = [
        "pop",
        "handwritten neat",
        "maru sans",
        "bold sans",
        "sans",
        "light sans",
        "bold serif",
        "serif",
        "light serif",
        "brush pop",
        "brush playful",
        "brush",
        "handwritten simple",
    ]

    # 学習用パラメータ / training parameters
    BATCH_SIZE = 64  # 引数で指定したいけど今のところここに書くしかない / We want to specify it as an argument, but for now we have to write it here.
    SEED = 42

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.width == self.height, "width and height must be the same"
        print(f"JoyoKanjiDataset: width={self.width}, height={self.height}, max_token_length={self.max_token_length}")

        # read letters
        letters = set()
        with open(JoyoKanjiDataset.LETTERS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                for c in line:
                    letters.add(c)
        self._letters = sorted(list(letters))

        # prepare fonts
        self._fonts: ImageFont = []
        self._font_offsets = []
        for font_file in JoyoKanjiDataset.FONT_FILES:
            # フォントファイルが存在するか確認し読み込む / Check if the font file exists and read it
            assert os.path.exists(font_file), f"font file {font_file} does not exist"
            font = ImageFont.truetype(font_file, self.width * 4 // 5)
            self._fonts.append(font)

            # 文字を上下中央に配置するためのオフセットを計算 / Calculate the offset to place the character in the center vertically
            # ascent, descent = font.getmetrics()
            # (width, baseline), (offset_x, offset_y) = font.font.getsize(text)
            _, (_, offset_y) = font.font.getsize("亜")
            self._font_offsets.append(offset_y)

        # シャッフルのためのインデックスを用意 / prepare index for shuffle
        self._indices = list(range(len(self._letters)))

        # メタデータを用意 / prepare metadata
        self.img_count = len(self._letters)
        self.num_train_images = self.img_count
        self.batch_size = JoyoKanjiDataset.BATCH_SIZE

        # debug_datasetで使われるメタデータを用意 / prepare metadata used in debug_dataset
        # debug_datasetを使わない場合はたぶん不要 / probably not needed if you don't use debug_dataset
        self.image_data = {}
        for letter in self._letters:
            self.image_data[letter] = ImageInfo(letter, 1, letter, False, "dummy")

        print(f"JoyoKanjiDataset: {len(self._letters)} letters, {len(self._fonts)} fonts")

    def __len__(self):
        # バッチサイズに切り上げる / round up to batch size
        return (len(self._letters) + JoyoKanjiDataset.BATCH_SIZE - 1) // JoyoKanjiDataset.BATCH_SIZE

    def set_current_epoch(self, epoch):
        # この関数は各エポックの最初に呼ばれる / This function is called at the beginning of each epoch
        # Datasetがマルチプロセスで動くので、ランダムシードを設定してシャッフルすることで、各プロセスで同じ順番でデータを処理するようにする
        # Dataset runs in multiprocess, so set a random seed and shuffle to process the data in the same order in each process
        self._indices.sort()
        rnd_state = random.getstate()
        random.seed(JoyoKanjiDataset.SEED + epoch)
        random.shuffle(self._indices)
        random.setstate(rnd_state)

        return super().set_current_epoch(epoch)

    def __getitem__(self, index):
        image_keys = []
        images = []
        captions = []
        input_ids_list = []

        for i in range(JoyoKanjiDataset.BATCH_SIZE):
            letter_index = index * JoyoKanjiDataset.BATCH_SIZE + i

            # 最後のバッチはBATCH_SIZEに満たない場合があるので、ランダムに文字を選ぶ / The last batch may not be full, so choose a character at random
            if letter_index >= len(self._letters):
                letter_index = random.randint(0, len(self._letters) - 1)
            letter_index = self._indices[letter_index]

            letter = self._letters[letter_index]

            # ランダムにフォントを選ぶ / Choose a font at random
            font_index = random.randint(0, len(self._fonts) - 1)
            font = self._fonts[font_index]

            # 画像を生成 / Generate image
            img = Image.new("RGB", (self.width, self.height), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)

            # テキストサイズを計算して中央に描画 / Calculate text size and draw in the center
            text_size = draw.textsize(letter, font=font)
            y_offset = self._font_offsets[font_index]
            draw.text(
                ((self.width - text_size[0]) // 2, (self.height - text_size[1] + y_offset) // 2 - y_offset),
                letter,
                font=font,
                fill=(0, 0, 0),
            )

            # キャプションを生成 / Generate caption
            cap_fragments = []
            preposition = "by" if random.random() < 0.33 else ("with" if random.random() < 0.5 else "in")
            article = "" if random.random() < 0.5 else "the"
            cap_fragments.append(f"{preposition} {JoyoKanjiDataset.FONT_NAMES[font_index]}")
            cap_fragments.append(f"{article} letter {letter}") # characterのほうがいいかも…… / character might be better...

            # キャプションをシャッフルして連結する / Shuffle and concatenate captions
            random.shuffle(cap_fragments)
            caption = ", ".join(cap_fragments)

            # Textual Inversionの場合は、captionにtoken_stringを含んでおき、以下を実行すると行けるはず
            # For Textual Inversion, include token_string in caption and execute the following
            # for str_from, str_to in self.replacements.items():
            #     if str_from == "":
            #         # replace all
            #         if type(str_to) == list:
            #             caption = random.choice(str_to)
            #         else:
            #             caption = str_to
            #     else:
            #         caption = caption.replace(str_from, str_to)

            # トークンIDを取得 / Get token IDs
            input_ids = self.get_input_ids(caption)

            # バッチに追加 / Add to batch
            img_tensor = self.image_transforms(img)
            image_keys.append(letter)
            images.append(img_tensor)
            captions.append(caption)
            input_ids_list.append(input_ids)

        # ListからTensorに変換して返す / Convert from list to Tensor and return
        images = torch.stack(images, dim=0)
        input_ids_list = torch.stack(input_ids_list, dim=0)
        example = {
            "images": images,
            "input_ids": input_ids_list,
            "captions": captions,
            "latents": None,
            "image_keys": image_keys, # for debug_dataset
            # 1.0がデフォルトだが他の値を設定して重みづけを変えることもできる。数はバッチサイズと同じにすること
            # 1.0 is the default, but you can also set other values to change the weighting. The number should be the same as the batch size.
            "loss_weights": torch.ones(JoyoKanjiDataset.BATCH_SIZE, dtype=torch.float32),  
        }
        return example
