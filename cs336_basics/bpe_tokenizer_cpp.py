"""
C++ optimized BPE tokenizer components using cppyy
"""

import cppyy
from typing import Dict, List, Tuple

# Define the C++ code
cpp_code = """
#include <vector>
#include <unordered_map>
#include <queue>
#include <tuple>
#include <utility>
#include <stdexcept>
#include <optional>
#include <string>

struct MergeRule {
    int token1;
    int token2;
    int priority;
    int replacement;
};

struct QueueItem {
    int priority;
    int replacement;
    int i;
    int j;
    int token1;
    int token2;
    
    // For min-heap behavior
    bool operator>(const QueueItem& other) const {
        if (priority != other.priority) return priority > other.priority;
        return replacement > other.replacement;
    }
};

struct MergeInfo {
    int priority;
    int replacement;
};

struct VocabItem {
    int token_id;
    std::string byte_seq;
};

// Custom hash for std::pair<int, int>
struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

class OptimizedWorkQueue {
private:
    std::unordered_map<std::pair<int, int>, MergeInfo, PairHash> token_merges_;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue_;
    
public:
    OptimizedWorkQueue(const std::vector<MergeRule>& merges) {
        for (const auto& merge : merges) {
            std::pair<int, int> key = std::make_pair(merge.token1, merge.token2);
            token_merges_[key] = MergeInfo{.priority = merge.priority, .replacement = merge.replacement};
        }
    }
    
    void maybe_push(int i, int j, const std::vector<int>& tokens) {
        if (i >= tokens.size() || j >= tokens.size()) {
            throw std::out_of_range("Index out of range in maybe_push: i=" + std::to_string(i) + 
                                  ", j=" + std::to_string(j) + ", tokens.size()=" + std::to_string(tokens.size()));
        }
        
        auto it = token_merges_.find(std::make_pair(tokens[i], tokens[j]));
        if (it == token_merges_.end()) return;
        
        const MergeInfo& info = it->second;
        queue_.push(QueueItem{
            .priority = info.priority,
            .replacement = info.replacement,
            .i = i,
            .j = j,
            .token1 = tokens[i],
            .token2 = tokens[j]
        });
    }
    
    std::optional<QueueItem> pop() {
        if (queue_.empty()) {
            return std::nullopt;
        }
        
        QueueItem item = queue_.top();
        queue_.pop();
        return item;
    }
    
    void clear() {
        while (!queue_.empty()) {
            queue_.pop();
        }
    }
    
    bool empty() const {
        return queue_.empty();
    }
};

class OptimizedBPEEncoder {
private:
    std::unordered_map<int, std::string> vocab_;
    std::unordered_map<std::string, int> inverse_vocab_;
    OptimizedWorkQueue queue_;
    
public:
    OptimizedBPEEncoder(const std::vector<VocabItem>& vocab,
                       const std::vector<MergeRule>& merges) 
        : queue_(merges) {
        
        for (const auto& item : vocab) {
            vocab_[item.token_id] = item.byte_seq;
            inverse_vocab_[item.byte_seq] = item.token_id;
        }
    }
    
    std::vector<int> encode_pretoken(const std::string& pretoken) {
        // Convert bytes to initial tokens (identity mapping)
        std::vector<int> tokens;
        tokens.reserve(pretoken.size());
        
        std::string inverse_token(1, '\\0');
        for (unsigned char byte : pretoken) {
            inverse_token[0] = byte;
            auto it = inverse_vocab_.find(inverse_token);
            if (it == inverse_vocab_.end()) {
                throw std::out_of_range("Invalid vocabulary element: " + std::to_string(byte));
            }
            tokens.push_back(it->second);
        }
        
        // Clear the queue for this encoding
        queue_.clear();
        
        // Initialize queue with adjacent pairs
        for (int i = 0; i < (int)tokens.size() - 1; ++i) {
            queue_.maybe_push(i, i + 1, tokens);
        }
        
        // Process merges
        while (true) {
            std::optional<QueueItem> result = queue_.pop();
            if (!result.has_value()) break;  // Queue is empty
            
            QueueItem item = result.value();
            int priority = item.priority;
            int replacement = item.replacement;
            int i = item.i;
            int i_next = item.j;
            int token1 = item.token1;
            int token2 = item.token2;
            
            // Check if tokens are still valid at these positions
            if (i >= tokens.size() || i_next >= tokens.size()) {
                throw std::out_of_range("Token positions out of range during merge: i=" + std::to_string(i) + 
                                      ", i_next=" + std::to_string(i_next) + ", tokens.size()=" + std::to_string(tokens.size()));
            }
            
            if (tokens[i] != token1 || tokens[i_next] != token2) {
                continue;
            }
            
            // Perform the merge
            tokens[i] = replacement;
            tokens[i_next] = -1;  // Mark as deleted
            
            // Look for new merge opportunities to the left
            int j = i - 1;
            while (j >= 0) {
                if (tokens[j] == -1) {
                    j--;
                    continue;
                }
                queue_.maybe_push(j, i, tokens);
                break;
            }
            
            // Look for new merge opportunities to the right
            j = i + 1;
            while (j < (int)tokens.size()) {
                if (tokens[j] == -1) {
                    j++;
                    continue;
                }
                queue_.maybe_push(i, j, tokens);
                break;
            }
        }
        
        // Filter out deleted tokens
        std::vector<int> result;
        result.reserve(tokens.size());
        for (int token : tokens) {
            if (token >= 0) {
                result.push_back(token);
            }
        }
        
        return result;
    }

    std::vector<int> encode_pretokens(const std::vector<std::string>& pretokens) {
        std::vector<int> return_tokens;
        for (auto& pretoken : pretokens) {
            auto tokens = encode_pretoken(pretoken);
            return_tokens.insert(return_tokens.end(), tokens.begin(), tokens.end());
        }
        return return_tokens;
    }
    
    std::string decode_tokens(const std::vector<int>& tokens) const {
        std::string result;
        for (int token : tokens) {
            auto it = vocab_.find(token);
            if (it != vocab_.end()) {
                const auto& bytes = it->second;
                result.insert(result.end(), bytes.begin(), bytes.end());
            }
        }
        return result;
    }
};
"""

# Compile the C++ code
cppyy.cppdef(cpp_code)


class OptimizedBPEEncoder:
    """Python wrapper for the optimized C++ BPE encoder"""

    def __init__(
        self,
        vocab: Dict[int, bytes],
        token_merges: Dict[Tuple[int, int], Tuple[int, int]],
    ):
        self._vocab = vocab.copy()

        # Convert vocab to C++ format
        vocab_list = []
        for token_id, byte_seq in vocab.items():
            vocab_list.append(cppyy.gbl.VocabItem(token_id=token_id, byte_seq=byte_seq))

        # Convert merges to C++ format
        merge_list = []
        for (token1, token2), (priority, replacement) in token_merges.items():
            merge_list.append(
                cppyy.gbl.MergeRule(
                    token1=token1,
                    token2=token2,
                    priority=priority,
                    replacement=replacement,
                )
            )

        self._cpp_encoder = cppyy.gbl.OptimizedBPEEncoder(vocab_list, merge_list)

    def encode_pretoken(self, pretoken: bytes) -> List[int]:
        result = self._cpp_encoder.encode_pretoken(pretoken)

        # if __debug__:
        #     decoded_bytes = bytes(self._cpp_encoder.decode_tokens(result))
        #     assert decoded_bytes == pretoken, f"{decoded_bytes} != {pretoken}"

        return result

    def encode_pretokens(self, pretokens: list[bytes]) -> List[int]:
        result = self._cpp_encoder.encode_pretokens(pretokens)

        # if __debug__:
        #     decoded_bytes = bytes(self._cpp_encoder.decode_tokens(result))
        #     assert decoded_bytes == pretoken, f"{decoded_bytes} != {pretoken}"

        return result
