from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 args,
                 pid_dict,
                 transform=None,
                 transform_with_mask=None,
                 text_length: int = 77,
                 truncate: bool = True
                 ):
        self.args = args
        self.pid_dict = pid_dict
        self.dataset = dataset
        self.transform = transform
        self.transform_with_mask = transform_with_mask
        self.text_length = text_length
        self.truncate = truncate
        # targets = np.asarray([s[0] for s in self.dataset])
        # self.targets = targets

        self.tokenizer = SimpleTokenizer()

        # 男性词汇数组
        self.male_words = ['he', 'boy', 'man', 'Mr.', 'father', 'male', 'his', 'gentleman', 'masculine', 'guy']

        # 女性词汇数组
        self.female_words = ['she', 'girl', 'woman', 'Mrs.', 'Ms.', 'mother', 'female', 'her', 'lady', 'feminine', 'miss']

    def __len__(self):
        return len(self.dataset)

    def get_random_indexes(self, different_image_id_indexes, num_samples=8):
        if len(different_image_id_indexes) < num_samples:
            additional_indexes = random.choices(different_image_id_indexes,
                                                k=num_samples - len(different_image_id_indexes))
            random_indexes = different_image_id_indexes + additional_indexes
        else:
            random_indexes = random.sample(different_image_id_indexes, num_samples)

        return random_indexes

    def get_random_index(self, current_index):
        current_pid = self.dataset[current_index][0]
        current_image_id = self.dataset[current_index][1]
        possible_indexes = self.pid_dict[current_pid]

        different_image_id_indexes = [idx for idx in possible_indexes if self.dataset[idx][1] != current_image_id]

        if not different_image_id_indexes:
            different_id_indexes = [idx for idx in possible_indexes]
            return self.get_random_indexes(different_id_indexes, 4)

        # 随机选择
        # return random.choice(different_image_id_indexes)
        return self.get_random_indexes(different_image_id_indexes, 4)

    def determine_gender(self, text):
        # 初始化男性和女性词汇计数
        male_count = 0
        female_count = 0

        # 将文本分词
        words = re.findall(r'\b\w+\b', text.lower())

        # 统计男性和女性词汇的数量
        for word in words:
            if word in self.male_words:
                male_count += 1
            elif word in self.female_words:
                female_count += 1

        # 根据词汇数量判断性别
        if male_count > female_count:
            return 0
        elif female_count > male_count:
            return 1
        else:
            return 2

    # # 示例文本
    # text = 'The girl has long black hair with bangs. Her hair is pulled forward on her shoulders. She is wearing a shirt sleeve white and mint green shirt with dark coloured shorts. She is carrying something white in her left hand.'
    # # 判断性别
    # gender = determine_gender(text)
    # print("该文本描述的性别为:", gender)
    #
    # text = 'The woman has brown greying, curly neck length hair, is wearing a floral hat, above the ankle length long-sleeved dress decorated with numerous purple, pink, and white floral pattern, and brown slippers.'
    # # 处理数据
    # output = determine_gender(text)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]

        img = read_image(img_path)
        # print(img_path)
        m_img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.transform_with_mask is not None:
            m_img = self.transform_with_mask(m_img)

        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length,
                                  truncate=self.truncate)
        m_caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length,
                                  truncate=self.truncate)
        caption_tokens, _ = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        # m_caption_tokens, mlm_labels = self.random_masked_tokens_and_labels(m_caption_tokens.cpu().numpy())

        gender = self.determine_gender(caption)

        # caption_token = mlm_tokens.clone()
        #
        # mask_tokens, _ = self._build_random_masked_tokens_and_labels(mlm_tokens.cpu().numpy())

        new_index = self.get_random_index(index)

        if not isinstance(new_index, list):
            print(new_index)
            new_index = [new_index]

        ps_imgs = []
        ps_m_imgs = []

        ps_caption_tokens = []
        ps_mask_tokens = []
        for i in new_index:
            ps_pid, ps_image_id, ps_img_path, ps_caption = self.dataset[i]
            ps_img = read_image(ps_img_path)
            ps_m_img = read_image(ps_img_path)
            if self.transform is not None:
                ps_img = self.transform(ps_img)
                ps_imgs.append(ps_img)

            if self.transform_with_mask is not None:
                ps_m_img = self.transform_with_mask(ps_m_img)
                ps_m_imgs.append(ps_m_img)

            ps_caption_token = tokenize(ps_caption, tokenizer=self.tokenizer, text_length=self.text_length,
                                         truncate=self.truncate)
            m_ps_caption_token = tokenize(ps_caption, tokenizer=self.tokenizer, text_length=self.text_length,
                                          truncate=self.truncate)

            ps_caption_token, ps_mlm_labels = self._build_random_masked_tokens_and_labels(
                ps_caption_token.cpu().numpy())

            ps_caption_tokens.append(ps_caption_token)

            # ps_mask_token, _ = self.random_masked_tokens_and_labels(m_ps_caption_token.cpu().numpy())
            ps_mask_tokens.append(m_ps_caption_token)

        ps_imgs = torch.stack(ps_imgs)
        ps_m_imgs = torch.stack(ps_m_imgs)
        ps_caption_tokens = torch.stack(ps_caption_tokens)
        ps_mask_tokens = torch.stack(ps_mask_tokens)



        ret = {
            'pids': pid,
            'gender': gender,
            'image_ids': image_id,
            'images': img,
            'm_images': m_img,
            'caption_ids': caption_tokens,
            'mlm_ids': m_caption_tokens,
            # 'mlm_labels': mlm_labels,
            'ps_images': ps_imgs,
            'ps_m_images': ps_m_imgs,
            'ps_caption_ids': ps_caption_tokens,
            'ps_mlm_ids': ps_mask_tokens,
        }

        return ret

    def random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < self.args.t_mask:
                    prob /= self.args.t_mask

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.35:
                    prob /= 0.35

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)