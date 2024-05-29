
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_openai import OpenAI
import pandas as pd

class Cloth_Recommendations:
    def __init__(self,db_path):
        # OpenAI API 키 설정
        api_key = 'your_api'

        # OpenAI 임베딩 생성
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # CSV 파일 불러오기
        df = pd.read_csv(db_path)

        #문서 생성
        documents = []
        for idx in range(len(df)):
            item = df.iloc[idx]
            # page_content 검색에 사용되는 정보
            # metadata
            document = Document(
                page_content=f"{item['gender']} : {item['age']} : {item['category']} : {item['Price']} : {item['Like_Count']} likes : {item['color']} color : {item['season']} season : {item['usage']} usage : {item['Fit']} fit : {item['Material']} : {item['etc']}",
                metadata={
                    "Product_Name": item['Product_Name'],
                    "Product_Link": item['Product_Link'],
                    "Image_Link": item['Image_Link'],
                    "Brand_Name": item['Brand_Name'],
                    "Product_id": int(item['id']),
                    "prompt": item['prompt']
                }
            )
            documents.append(document)

        # ChromaDB에 데이터 저장
        self.vectorstore = Chroma.from_documents(documents, self.embeddings, persist_directory="../chroma_db")
    def Recommendations(self, text, num_recommendations=5):
        # 사용자의 질문을 받아 임베딩 생성
        query_text = f"<strong>{text}</strong>."
        # print(type(query_text))
        query_embedding = self.embeddings.embed_query(query_text)
        # 임베딩을 기반으로 ChromaDB에서 가장 유사한 아이템 검색
        results = self.vectorstore.similarity_search_by_vector(embedding=query_embedding,k= num_recommendations)
        #
        # # 검색 결과 반환
        # 아이디, 제품 링크, 이미지 링크만 추출
        # print(results)
        print(results)
        extracted_results = [(doc.metadata['Product_id'], doc.metadata['Image_Link'], doc.metadata['Product_Link'],
                              doc.metadata['prompt']) for
                             doc in results]
        extracted_results2 = [(doc.metadata['Product_id'], doc.metadata['Image_Link']) for
                             doc in results]
        self.vectorstore.delete_collection()
        # return  results
        return extracted_results , extracted_results2

# # 사용 예시
# age = 20
# gender = 'male'
#
# # 색상 ,성별,이름
# # 핏, 소재,계절
# text = "blue color"
# #
# # #
# #
# #
# # print(type(text))
#
# # # 추천 갯수
# num_recommendations = 5
# # Cloth_Recommendations 객체 생성
# cloth_re = Cloth_Recommendations('../db/cloth(2).csv')
#
# # 실행
# cloth = cloth_re.Recommendations( text, num_recommendations)
# #
# # # 결과 예시
# # # [(id,Product_Link,Image_Link), .....]
# # # [(114, 'https://www.musinsa.com/app/goods/1422202', 'https://image.msscdn.net/images/goods_img/20200427/1422202/1422202_1_500.jpg'),....]
# #
# #
# # print(cloth)
# print(type(cloth))