/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <utf8proc.h>

#include <algorithm>
#include <codecvt>
#include <exception>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "tokenizer.h"

namespace tokenizer {

using std::bad_cast;
using std::codecvt_utf8;
using std::cout;
using std::endl;
using std::exception;
using std::ifstream;
using std::min;
using std::runtime_error;
using std::unordered_map;
using std::unordered_set;
using std::shared_ptr;
using std::string;
using std::vector;
using std::wstring;
using std::wstring_convert;

std::wstring_convert<std::codecvt_utf8<wchar_t>> kConverter;
const wstring kStripChars = L" \t\n\r\v\f";

void ConvertStrToWstr(const std::string& src, std::wstring* res) {
  *res = kConverter.from_bytes(src);
}

void ConvertWstrToStr(const std::wstring& src, std::string* res) {
  *res = kConverter.to_bytes(src);
}

void NormalizeNfd(const std::string& s, std::string* ret) {
  *ret = "";
  char* result = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())));
  if (result) {
    *ret = std::string(result);
    free(result);
  }
}

bool IsControl(const wchar_t& ch) {
  if (ch == L'\t' || ch == L'\n' || ch == L'\r') return false;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF) return true;
  return false;
}

bool IsWhiteSpace(const wchar_t& ch) {
  if (ch == L' ' || ch == L'\t' || ch == L'\n' || ch == L'\r') return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_ZS) return true;
  return false;
}

bool IsPunctuation(const wchar_t& ch) {
  if ((ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) ||
      (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126))
    return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_PD || cat == UTF8PROC_CATEGORY_PS ||
      cat == UTF8PROC_CATEGORY_PE || cat == UTF8PROC_CATEGORY_PC ||
      cat == UTF8PROC_CATEGORY_PO  // sometimes ¶ belong SO
      ||
      cat == UTF8PROC_CATEGORY_PI || cat == UTF8PROC_CATEGORY_PF)
    return true;
  return false;
}

bool IsStripChar(const wchar_t& ch) {
  return kStripChars.find(ch) != wstring::npos;
}

void Strip(const wstring& text, wstring* ret) {
  *ret = text;
  if (ret->empty()) return;
  size_t pos = 0;
  while (pos < ret->size() && IsStripChar(ret->at(pos))) pos++;
  if (pos != 0) *ret = ret->substr(pos, ret->size() - pos);
  pos = ret->size() - 1;
  while (IsStripChar(ret->at(pos))) pos--;
  ret->substr(0, pos + 1);
}

vector<wstring> Split(const wstring& text) {
  vector<wstring> result;
  boost::split(result, text, boost::is_any_of(kStripChars));
  return result;
}

void Split(const wstring& text, vector<wstring>* result) {
  // vector<wstring> result;
  boost::split(*result, text, boost::is_any_of(kStripChars));
  // return result;
}

void WhiteSpaceTokenize(const wstring& text, vector<wstring>* res) {
  wstring stext;
  Strip(text, &stext);
  if (stext.empty()) {
    return;
  } else {
    Split(text, res);
  }
}

void ToLower(const wstring& s, wstring* res) {
  res->clear();
  res->resize(s.size());
  for (size_t i = 0; i < s.size(); i++) {
    res->at(i) = utf8proc_tolower(s[i]);
  }
}

void LoadVocab(const std::string& vocabFile, Vocab* vocab) {
  size_t index = 0;
  std::ifstream ifs(vocabFile, std::ifstream::in);
  if (!ifs) {
    throw std::runtime_error("open file failed");
  }
  std::string line;
  while (getline(ifs, line)) {
    std::wstring token;
    ConvertStrToWstr(line, &token);
    if (token.empty()) break;
    (*vocab)[token] = index;
    index++;
  }
}

BasicTokenizer::BasicTokenizer(bool do_lower_case /* = true */)
    : do_lower_case_(do_lower_case) {}

void BasicTokenizer::clean_text(const wstring& text, wstring* output) const {
  output->clear();
  wchar_t space_char = L' ';
  for (const wchar_t& cp : text) {
    if (cp == 0 || cp == 0xfffd || IsControl(cp)) continue;
    if (IsWhiteSpace(cp))
      output->push_back(std::move(space_char));
    else
      output->push_back(std::move(cp));
  }
}

bool BasicTokenizer::is_chinese_char(const wchar_t& ch) const {
  if ((ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF) ||
      (ch >= 0x20000 && ch <= 0x2A6DF) || (ch >= 0x2A700 && ch <= 0x2B73F) ||
      (ch >= 0x2B740 && ch <= 0x2B81F) || (ch >= 0x2B820 && ch <= 0x2CEAF) ||
      (ch >= 0xF900 && ch <= 0xFAFF) || (ch >= 0x2F800 && ch <= 0x2FA1F))
    return true;
  return false;
}

void BasicTokenizer::tokenize_chinese_chars(const wstring& text,
                                            wstring* output) const {
  wchar_t space_char = L' ';
  for (auto& ch : text) {
    if (is_chinese_char(ch)) {
      output->push_back(std::move(space_char));
      output->push_back(std::move(ch));
      output->push_back(std::move(space_char));
    } else {
      output->push_back(std::move(ch));
    }
  }
}

void BasicTokenizer::run_strip_accents(const wstring& text,
                                       wstring* output) const {
  // Strips accents from a piece of text.
  wstring unicode_text;
  try {
    string tmp, nor_tmp;
    ConvertWstrToStr(text, &tmp);
    NormalizeNfd(tmp, &nor_tmp);
    ConvertStrToWstr(nor_tmp, &unicode_text);
  } catch (bad_cast& e) {
    std::cout << e.what() << endl;
    *output = L"";
    return;
  }
  output->clear();
  for (auto& ch : unicode_text) {
    auto&& cat = utf8proc_category(ch);
    if (cat == UTF8PROC_CATEGORY_MN) continue;
    output->push_back(std::move(ch));
  }
}

void BasicTokenizer::run_split_on_punc(const wstring& text,
                                       vector<wstring>* output) const {
  output->clear();
  size_t i = 0;
  bool start_new_word = true;
  while (i < text.size()) {
    wchar_t ch = text[i];
    if (IsPunctuation(ch)) {
      output->emplace_back(wstring(&ch, 1));
      start_new_word = true;
    } else {
      if (start_new_word) output->emplace_back(wstring());
      start_new_word = false;
      output->at(output->size() - 1) += ch;
    }
    i++;
  }
}

void BasicTokenizer::Tokenize(const string& text, vector<wstring>* res) const {
  wstring tmp, c_text, unicode_text;
  ConvertStrToWstr(text, &tmp);
  clean_text(tmp, &c_text);
  tokenize_chinese_chars(c_text, &unicode_text);

  vector<wstring> original_tokens;
  WhiteSpaceTokenize(unicode_text, &original_tokens);

  vector<wstring> split_tokens;
  for (wstring& token : original_tokens) {
    if (do_lower_case_) {
      tmp.clear();
      ToLower(token, &tmp);
      wstring stoken;
      run_strip_accents(tmp, &stoken);
    }
    vector<wstring> tokens;
    run_split_on_punc(token, &tokens);
    for (size_t i = 0; i < tokens.size(); ++i) {
      split_tokens.emplace_back(tokens[i]);
    }
  }
  WhiteSpaceTokenize(boost::join(split_tokens, L" "), res);
}

WordPieceTokenizer::WordPieceTokenizer(
    Vocab& vocab,
    const wstring& unk_token /* = L"[UNK]"*/,
    const size_t max_input_chars_per_word /* = 100 */)
    : vocab_(vocab),
      unk_token_(unk_token),
      max_input_chars_per_word_(max_input_chars_per_word) {}

void WordPieceTokenizer::Tokenize(const wstring& text,
                                  vector<wstring>* output_tokens) const {
  // vector<wstring> output_tokens;
  vector<wstring> tokens;
  WhiteSpaceTokenize(text, &tokens);
  for (auto& token : tokens) {
    if (token.size() > max_input_chars_per_word_) {
      output_tokens->emplace_back(unk_token_);
    }
    bool is_bad = false;
    size_t start = 0;
    vector<wstring> sub_tokens;
    while (start < token.size()) {
      size_t end = token.size();
      wstring cur_sub_str;
      bool has_cur_sub_str = false;
      while (start < end) {
        wstring substr = token.substr(start, end - start);
        if (start > 0) substr = L"##" + substr;
        if (vocab_.find(substr) != vocab_.end()) {
          cur_sub_str = substr;
          has_cur_sub_str = true;
          break;
        }
        end--;
      }
      if (!has_cur_sub_str) {
        is_bad = true;
        break;
      }
      sub_tokens.emplace_back(cur_sub_str);
      start = end;
    }
    if (is_bad) {
      output_tokens->emplace_back(unk_token_);
    } else {
      for (size_t i = 0; i < sub_tokens.size(); ++i)
        output_tokens->emplace_back(sub_tokens[i]);
    }
  }
}

BertTokenizer::BertTokenizer(Vocab& vocab,
                             const bool& do_lower_case /* = false */,
                             const wstring& unk_token /* = L"[UNK]" */,
                             const wstring& pad_token /* = L"[PAD]" */,
                             const wstring& cls_token /* = L"[CLS]" */,
                             const wstring& mask_token /* = L"[MASK]" */,
                             const wstring& sep_token /* = L"[SEP]" */,
                             const string& padding_site /* = "right" */)
    : do_lower_case_(do_lower_case),
      unk_token_(unk_token),
      pad_token_(pad_token),
      cls_token_(cls_token),
      mask_token_(mask_token),
      sep_token_(sep_token),
      padding_site_(padding_site),
      vocab_(vocab),
      basic_tokenizer_(do_lower_case_),
      word_piece_tokenizer_(vocab_, unk_token) {
  unk_token_id_ = vocab_[unk_token_];
  pad_token_id_ = vocab_[pad_token_];
  cls_token_id_ = vocab_[cls_token_];
  mask_token_id_ = vocab_[mask_token_];
  sep_token_id_ = vocab_[sep_token_];

  all_special_tokens_ = vector<wstring>(
      {unk_token_, pad_token_, cls_token_, mask_token_, sep_token_});
  all_special_token_ids_ = unordered_set<int64_t>({unk_token_id_,
                                                   pad_token_id_,
                                                   cls_token_id_,
                                                   mask_token_id_,
                                                   sep_token_id_});
}

void BertTokenizer::ConvertTokensToIds(const vector<wstring>& tokens,
                                       vector<int64_t>* token_ids) const {
  token_ids->clear();
  token_ids->resize(tokens.size());
  for (size_t i = 0; i < token_ids->size(); ++i) {
    auto iter = vocab_.find(tokens[i]);
    if (iter != vocab_.end()) {
      token_ids->at(i) = iter->second;
    } else {
      token_ids->at(i) = unk_token_id_;
    }
  }
}

void BertTokenizer::ConvertTokensToString(const vector<wstring>& tokens,
                                          string* res) const {
  ConvertWstrToStr(boost::join(tokens, L" "), res);
}

void BertTokenizer::Tokenize(const string& text,
                             vector<wstring>* split_tokens) const {
  vector<wstring> tokens;
  basic_tokenizer_.Tokenize(text, &tokens);
  for (auto& token : tokens) {
    vector<wstring> sub_tokens;
    word_piece_tokenizer_.Tokenize(token, &sub_tokens);
    for (size_t i = 0; i < sub_tokens.size(); ++i) {
      split_tokens->emplace_back(sub_tokens[i]);
    }
  }
}

void BertTokenizer::BuildInputsWithSpecialTokens(
    vector<int64_t>* inputs,
    const vector<int64_t>& token_ids_0,
    const vector<int64_t>& token_ids_1 /* = vector<int64_t>() */) const {
  if (token_ids_1.size() == 0) {
    inputs->clear();
    inputs->emplace_back(cls_token_id_);
    for (auto& token_id : token_ids_0) {
      inputs->emplace_back(token_id);
    }
    inputs->emplace_back(sep_token_id_);
  } else {
    inputs->clear();
    inputs->emplace_back(cls_token_id_);
    for (auto& token_id : token_ids_0) {
      inputs->emplace_back(token_id);
    }
    inputs->emplace_back(sep_token_id_);
    for (auto& token_id : token_ids_1) {
      inputs->emplace_back(token_id);
    }
  }
}

void BertTokenizer::GetSpecialTokensMask(
    vector<int64_t>* res,
    const vector<int64_t>& token_ids_0,
    const vector<int64_t>& token_ids_1 /* = vector<int64_t>() */,
    const bool already_has_special_tokens /* = false */) const {
  if (already_has_special_tokens) {
    if (token_ids_1.size() != 0) {
      throw runtime_error(
          "You should not supply a second sequence if the provided sequence of "
          "ids is already formatted with special tokens for the model.");
    }
    res->clear();
    res->resize(token_ids_0.size());
    for (size_t i = 0; i < res->size(); i++) {
      auto&& iter = std::find(all_special_token_ids_.begin(),
                              all_special_token_ids_.end(),
                              token_ids_0[i]);
      if (iter != all_special_token_ids_.end()) {
        res->at(i) = 1;
      } else {
        res->at(i) = 0;
      }
    }
    return;
  }

  if (token_ids_1.size() != 0) {
    res->clear();
    res->resize(3 + token_ids_0.size() + token_ids_1.size(), 0);
    res->at(0) = 1;
    res->at(token_ids_0.size() + 1) = 1;
    res->at(2 + token_ids_0.size() + token_ids_1.size()) = 1;
  } else {
    res->clear();
    res->resize(2 + token_ids_0.size(), 0);
    res->at(0) = 1;
    res->at(token_ids_0.size() + 1) = 1;
  }
}

int64_t BertTokenizer::GetNumSpecialTokensToAdd(const bool pair) const {
  if (pair) {
    return 3;
  } else {
    return 2;
  }
}

void BertTokenizer::CreateTokenTypeIdsFromSequences(
    vector<int64_t>* token_type_ids,
    const vector<int64_t>& token_ids_0,
    const vector<int64_t>& token_ids_1 /* = vector<int64_t>() */) const {
  if (token_ids_1.size() == 0) {
    vector<int64_t> tmp(token_ids_0.size() + 2, 0);
    token_type_ids->swap(tmp);
  } else {
    vector<int64_t> tmp(token_ids_0.size() + token_ids_1.size() + 3, 0);
    for (size_t i = token_ids_0.size() + 2; i < tmp.size(); i++) {
      tmp[i] = 1;
    }
    token_type_ids->swap(tmp);
  }
}

int BertTokenizer::TruncateSequence(
    vector<int64_t>* ids,
    vector<int64_t>* pair_ids,
    const size_t num_tokens_to_remove /* = 0 */,
    const string& truncation_strategy /* = "longest_first" */,
    const size_t stride /* = 0 */) const {
  if (truncation_strategy == "longest_first") {
    for (size_t i = 0; i < num_tokens_to_remove; i++) {
      if ((pair_ids->size() == 0) || (ids->size() > pair_ids->size())) {
        ids->pop_back();
      } else {
        pair_ids->pop_back();
      }
    }
  } else if (truncation_strategy == "only_first") {
    if (ids->size() > num_tokens_to_remove) {
      for (size_t i = 0; i < num_tokens_to_remove; i++) {
        ids->pop_back();
      }
    } else {
      cout << "We need to remove {num_tokens_to_remove} "
              "to truncate the input but the first sequence has a length "
           << ids->size() << ". Please select another truncation strategy than "
           << truncation_strategy
           << ", for instance \'longest_first\' or \'only_second\'." << endl;
      // Failed.
      return 0;
    }
  } else if (truncation_strategy == "only_second" && pair_ids->size() != 0) {
    if (pair_ids->size() > num_tokens_to_remove) {
      for (size_t i = 0; i < num_tokens_to_remove; i++) {
        pair_ids->pop_back();
      }
    } else {
      cout << "We need to remove " << num_tokens_to_remove
           << " to truncate the input but the second sequence has a length "
           << pair_ids->size()
           << ". Please select another truncation strategy than "
           << truncation_strategy
           << ", for instance \'longest_first\' or \'only_first\'." << endl;
      // Failed.
      return 0;
    }
  }
  // Successed.
  return 1;
}

void BertTokenizer::get_input_ids(const string& text,
                                  vector<int64_t>* token_ids) const {
  vector<wstring> tokens;
  Tokenize(text, &tokens);
  ConvertTokensToIds(tokens, token_ids);
}

int64_t BertTokenizer::GetClsTokenID() const { return cls_token_id_; }

int64_t BertTokenizer::GetSepTokenID() const { return sep_token_id_; }

int64_t BertTokenizer::GetUnkTokenID() const { return unk_token_id_; }

int64_t BertTokenizer::GetMaskTokenID() const { return mask_token_id_; }

int64_t BertTokenizer::GetPadTokenID() const { return pad_token_id_; }

int BertTokenizer::Encode(
    unordered_map<string, vector<int64_t>>* encoded_inputs,
    const string& text,
    const string& text_pair /* = "" */,
    const size_t max_seq_len /* = 0 */,
    bool pad_to_max_seq_len /* = false */,
    bool return_length /* = false */,
    bool return_token_type_ids /* = true */,
    bool return_position_ids /* = false */,
    bool return_attention_mask /* = false */,
    const string& truncation_strategy /* = "longest_first" */,
    bool return_overflowing_tokens /* = false */,
    bool return_special_tokens_mask /* = false */) const {
  vector<int64_t> ids;
  get_input_ids(text, &ids);
  vector<int64_t> pair_ids;
  if (text_pair != "") {
    get_input_ids(text_pair, &pair_ids);
  }

  bool pair = false;
  if (pair_ids.size() != 0) {
    pair = true;
  }

  size_t len_ids = ids.size();
  size_t len_pair_ids = pair_ids.size();

  // Truncation: Handle max sequence length
  // If max_seq_len == 0, then do nothing and keep the real length.
  // If max_seq_len > 0 and
  // all the input sequence len is over the max_seq_len,
  // then we truncate it.
  size_t total_len = len_ids + len_pair_ids + GetNumSpecialTokensToAdd(pair);
  if (max_seq_len > 0 && total_len > max_seq_len) {
    unordered_map<string, vector<int64_t>> res;
    auto status = TruncateSequence(
        &ids, &pair_ids, total_len - max_seq_len, truncation_strategy);
    if (status == 0) {
      return 0;
    }
    if (return_overflowing_tokens) {
      encoded_inputs->emplace("overflowing_token_ids",
                              res["overflowing_token_ids"]);
      vector<int64_t> num_truncated_tokens(1, total_len - max_seq_len);
      encoded_inputs->emplace("num_truncated_tokens", num_truncated_tokens);
    }
  }

  // Add special tokens
  vector<int64_t> sequence;
  BuildInputsWithSpecialTokens(&sequence, ids, pair_ids);
  size_t seq_len = sequence.size();
  vector<int64_t> token_type_ids;
  CreateTokenTypeIdsFromSequences(&token_type_ids, ids, pair_ids);

  // Build output dictionnary
  encoded_inputs->emplace("input_ids", sequence);
  if (return_token_type_ids) {
    encoded_inputs->emplace("token_type_ids", token_type_ids);
  }
  if (return_special_tokens_mask) {
    vector<int64_t> special_token_mask;
    GetSpecialTokensMask(&special_token_mask, ids, pair_ids);
    encoded_inputs->emplace("special_tokens_mask", special_token_mask);
  }
  if (return_length) {
    vector<int64_t> len(1, seq_len);
    encoded_inputs->emplace("seq_len", len);
  }

  // Check lengths
  if (max_seq_len > 0 && seq_len > max_seq_len) {
    cout << "There is something wrong with the input sequence length."
            " Please check it."
         << endl;
    // Failed.
    return 0;
  }

  // Padding
  bool needs_to_be_padded = false;
  if (pad_to_max_seq_len && max_seq_len > 0 && (seq_len < max_seq_len)) {
    needs_to_be_padded = true;
  }

  if (needs_to_be_padded) {
    int64_t difference = max_seq_len - seq_len;
    if (padding_site_ == "right") {
      if (return_attention_mask) {
        vector<int64_t> attention_mask(max_seq_len, 0);
        for (size_t i = 0; i < seq_len; i++) {
          attention_mask[i] = 1;
        }
        encoded_inputs->emplace("attention_mask", attention_mask);
      }

      size_t pad_start = max_seq_len - 1 - difference;
      if (return_token_type_ids) {
        encoded_inputs->at("token_type_ids").resize(max_seq_len);
        for (size_t i = max_seq_len - 1; i > pad_start; i--) {
          encoded_inputs->at("token_type_ids")[i] = pad_token_id_;
        }
      }

      if (return_special_tokens_mask) {
        encoded_inputs->at("special_tokens_mask").resize(max_seq_len);
        for (size_t i = max_seq_len - 1; i > pad_start; i--) {
          encoded_inputs->at("special_tokens_mask")[i] = 1;
        }
      }

      encoded_inputs->at("input_ids").resize(max_seq_len);
      for (size_t i = max_seq_len - 1; i > pad_start; i--) {
        encoded_inputs->at("input_ids")[i] = pad_token_id_;
      }
    } else if (padding_site_ == "left") {
      if (return_attention_mask) {
        vector<int64_t> attention_mask = vector<int64_t>(max_seq_len, 0);
        for (size_t i = difference; i < max_seq_len; i++) {
          attention_mask[i] = 1;
        }
      }

      if (return_token_type_ids) {
        vector<int64_t> tmp(max_seq_len, pad_token_id_);
        for (size_t i = difference; i < max_seq_len; i++) {
          tmp[i] = encoded_inputs->at("token_type_ids")[i - difference];
        }
        encoded_inputs->at("token_type_ids").swap(tmp);
      }

      if (return_special_tokens_mask) {
        vector<int64_t> tmp(max_seq_len, 1);
        for (size_t i = difference; i < max_seq_len; i++) {
          tmp[i] = encoded_inputs->at("special_tokens_mask")[i - difference];
        }
        encoded_inputs->emplace("special_tokens_mask", tmp);
      }

      vector<int64_t> tmp(max_seq_len, pad_token_id_);
      for (size_t i = difference; i < max_seq_len; i++) {
        tmp[i] = encoded_inputs->at("input_ids")[i - difference];
      }
      encoded_inputs->at("input_ids").swap(tmp);
    }
  } else {
    if (return_attention_mask) {
      vector<int64_t> tmp(encoded_inputs->at("input_ids").size(), 1);
      encoded_inputs->emplace("attention_mask", tmp);
    }
  }

  if (return_position_ids) {
    vector<int64_t> position_ids(encoded_inputs->at("input_ids").size(), 0);
    for (size_t i = 0; i < encoded_inputs->at("input_ids").size(); i++) {
      position_ids[i] = i;
    }
    encoded_inputs->emplace("position_ids", position_ids);
  }
  return 1;
}

int BertTokenizer::BatchEncode(
    vector<unordered_map<string, vector<int64_t>>>* batch_encode_inputs,
    const vector<string>& batch_text,
    const vector<string>& batch_text_pair /* = vector<string>() */,
    bool is_split_into_words /* = false */,
    const size_t max_seq_len /* = 0 */,
    bool pad_to_max_seq_len /* = false */,
    bool return_length /* = false */,
    bool return_token_type_ids /* = true */,
    bool return_position_ids /* = false */,
    bool return_attention_mask /* = false */,
    const string& truncation_strategy /* = "longest_first" */,
    const size_t stride /* = 0 */,
    bool return_overflowing_tokens /* = false */,
    bool return_special_tokens_mask /* = false */) const {
  bool has_text_pair = false;
  if (batch_text_pair.size() != 0) {
    has_text_pair = true;
  }

  size_t batch_size = batch_text.size();
  batch_encode_inputs->clear();
  batch_encode_inputs->resize(batch_size);
  for (size_t i = 0; i < batch_size; i++) {
    if (stride > 0 && has_text_pair) {
      // TODO(Steffy-zxf): add processing for qa-task.
      cout << "Tokenizer op to precoss QA task data needs to be done." << endl;
      return 0;
    } else if (has_text_pair) {
      unordered_map<string, vector<int64_t>> res;
      auto status = Encode(&res,
                           batch_text[i],
                           batch_text_pair[i],
                           max_seq_len,
                           pad_to_max_seq_len,
                           return_length,
                           return_token_type_ids,
                           return_position_ids,
                           return_attention_mask,
                           truncation_strategy,
                           return_overflowing_tokens,
                           return_special_tokens_mask);
      if (status) {
        batch_encode_inputs->at(i) = std::move(res);
      } else {
        return 0;
      }
    } else {
      unordered_map<string, vector<int64_t>> res;
      auto status = Encode(&res,
                           batch_text[i],
                           {},
                           max_seq_len,
                           pad_to_max_seq_len,
                           return_length,
                           return_token_type_ids,
                           return_position_ids,
                           return_attention_mask,
                           truncation_strategy,
                           return_overflowing_tokens,
                           return_special_tokens_mask);
      if (status) {
        batch_encode_inputs->at(i) = std::move(res);
      } else {
        return 0;
      }
    }
  }
  // Successed.
  return 1;
}
};
