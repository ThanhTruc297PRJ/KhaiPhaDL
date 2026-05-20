import yake
from pyvi import ViTokenizer
import urllib.request
stopword = [
    "và", "là", "có", "cho", "của", "với", "trong", "một", "những",
    "các", "được", "này", "rất", "khá", "hơi", "thì", "mà", "để",
    "tôi", "sách", "nội", "dung", "phần", "chủ", "đề", "đọc", "học",
    "người", "nhiều", "còn", "nên", "về", "vào", "khi", "sau", "trên",
    "nếu", "như", "đã", "đang", "sẽ", "từ", "ra", "lại", "hơn", "nội dung"
]
def extract_text(text_data):
    # Lấy danh sách stopwords tiếng Việt (hoặc bạn có thể tự tạo list của riêng mình)
    # url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
    # response = urllib.request.urlopen(url)
    # vi_stopwords = response.read().decode('utf-8').split('\n')
    # print(vi_stopwords)
    text = text_data

    # Tách từ
    tokenized_text = ViTokenizer.tokenize(text)

    # Cấu hình YAKE
    custom_kw_extractor = yake.KeywordExtractor(
        lan="vi", 
        n=2,              # Độ dài tối đa của từ khóa (2 từ)
        dedupLim=0.9,     # Loại bỏ các từ khóa quá giống nhau
        top=8,            # Lấy 5 từ khóa
        features=None, 
        stopwords=stopword
    )

    keywords = custom_kw_extractor.extract_keywords(tokenized_text)
    return keywords
    # for kw, score in keywords:
    #     # Điểm của YAKE càng THẤP thì từ khóa càng quan trọng
    #     print(f"Từ khóa: '{kw.replace('_', ' ')}' - Điểm: {score:.4f}")