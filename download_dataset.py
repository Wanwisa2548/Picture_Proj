from icrawler.builtin import BingImageCrawler

# กำหนด Keyword ที่หลากหลายรวมถึงสภาพแสงต่างๆ
emotions_config = {
    "happy": ["happy face person bright light", "happy face person dim light", "happy face person night portrait"],
    "sad": ["sad face person shadows", "sad face person dark room", "sad face person low light"],
    "angry": ["angry face person dark background", "angry face person dramatic lighting", "angry face person dim lighting"],
    "neutral": ["neutral face person dark environment", "neutral face person low key lighting", "neutral face person indoor dark"]
}
for folder, keywords in emotions_config.items():
    print(f"Downloading {folder} images...")
    
    # รวมรูปจากหลาย Keyword ไว้ในโฟลเดอร์เดียวกัน
    for keyword in keywords:
        print(f"  - Searching for: {keyword}")
        crawler = BingImageCrawler(
            storage={"root_dir": f"dataset/{folder}"}
        )
        crawler.crawl(
            keyword=keyword,
            max_num=200, # กำหนดเป็น 200 ต่อ 1 คำค้นหา (รวมกัน 3 คำ = 600 รูปพอดี)
            filters={"size": "medium"}
        )

print("Download complete!")