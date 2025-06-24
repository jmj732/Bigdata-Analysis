import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS용
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """
    부산과 인천의 도로 및 교통사고 데이터를 로드하고 전처리합니다.
    """
    print("데이터 로딩 및 전처리 시작...")
    
    # 1. 부산 도로 데이터 로드 및 전처리
    print("부산 도로 데이터 처리 중...")
    busan_road = pd.read_csv("busan_road.csv", encoding="cp949")
    
    # 부산 도로 데이터 컬럼명 확인 및 정리
    print(f"부산 도로 데이터 컬럼: {list(busan_road.columns)}")
    print(f"부산 도로 데이터 형태: {busan_road.shape}")
    
    # 부산 도로 데이터 전처리
    busan_road_processed = busan_road.copy()
    
    # 숫자 컬럼들을 float로 변환
    numeric_columns = ['연장', '폭']
    for col in numeric_columns:
        if col in busan_road_processed.columns:
            busan_road_processed[col] = pd.to_numeric(busan_road_processed[col], errors='coerce')
    
    # 면적 계산 (연장 * 폭)
    busan_road_processed['면적'] = busan_road_processed['연장'] * busan_road_processed['폭']
    
    # 부산 도로 데이터 선택 및 컬럼명 통일
    busan_road_selected = busan_road_processed[['시도명', '시군구명', '도로위계', '도로명', '종속구분', '연장', '폭', '면적']].copy()
    busan_road_selected['데이터_출처'] = '부산'
    
    # 2. 인천 도로 데이터 로드 및 전처리
    print("인천 도로 데이터 처리 중...")
    inchun_road = pd.read_csv("data/Inchun_road.csv", encoding="cp949")
    
    # 인천 도로 데이터 컬럼명 확인 및 정리
    print(f"인천 도로 데이터 컬럼: {list(inchun_road.columns)}")
    print(f"인천 도로 데이터 형태: {inchun_road.shape}")
    
    # 인천 도로 데이터 전처리
    inchun_road_processed = inchun_road.copy()
    
    # 숫자 컬럼들을 float로 변환 (실제 컬럼명 사용)
    numeric_columns_inchun = ['폭원(미터)', '연장(미터)', '면적(제곱미터)']
    for col in numeric_columns_inchun:
        if col in inchun_road_processed.columns:
            inchun_road_processed[col] = pd.to_numeric(inchun_road_processed[col], errors='coerce')
    
    # 인천 도로 데이터 선택 및 컬럼명 통일
    inchun_road_selected = inchun_road_processed[['도로명(노선수)', '연장(미터)', '폭원(미터)', '면적(제곱미터)']].copy()
    inchun_road_selected.columns = ['도로명', '연장', '폭', '면적']
    inchun_road_selected['시도명'] = '인천광역시'
    inchun_road_selected['시군구명'] = ''  # 인천 데이터에는 시군구 정보가 없음
    inchun_road_selected['도로위계'] = ''  # 인천 데이터에는 도로위계 정보가 없음
    inchun_road_selected['종속구분'] = ''  # 인천 데이터에는 종속구분 정보가 없음
    inchun_road_selected['데이터_출처'] = '인천'
    
    # 3. 부산 교통사고 데이터 로드 및 전처리
    print("부산 교통사고 데이터 처리 중...")
    busan_accident = pd.read_csv("data/busan_accident.csv", encoding="cp949")
    
    print(f"부산 교통사고 데이터 컬럼: {list(busan_accident.columns)}")
    print(f"부산 교통사고 데이터 형태: {busan_accident.shape}")
    
    # 부산 교통사고 데이터 전처리
    busan_accident_processed = busan_accident.copy()
    
    # 숫자 컬럼들을 int로 변환
    numeric_columns_accident = ['발생건수', '사망자 수', '부상자 수']
    for col in numeric_columns_accident:
        if col in busan_accident_processed.columns:
            busan_accident_processed[col] = pd.to_numeric(busan_accident_processed[col], errors='coerce')
    
    busan_accident_processed['시도명'] = '부산광역시'
    busan_accident_processed['데이터_출처'] = '부산'
    
    # 4. 인천 교통사고 데이터 로드 및 전처리
    print("인천 교통사고 데이터 처리 중...")
    inchun_accident = pd.read_csv("data/Inchun_accident.csv", encoding="cp949")
    
    print(f"인천 교통사고 데이터 컬럼: {list(inchun_accident.columns)}")
    print(f"인천 교통사고 데이터 형태: {inchun_accident.shape}")
    
    # 인천 교통사고 데이터 전처리
    inchun_accident_processed = inchun_accident.copy()
    
    # 숫자 컬럼들을 float/int로 변환 (실제 컬럼명 사용)
    inchun_accident_processed['교통사고발생건수(건)'] = pd.to_numeric(inchun_accident_processed['교통사고발생건수(건)'], errors='coerce')
    inchun_accident_processed['자동차등록대수(대)'] = pd.to_numeric(inchun_accident_processed['자동차등록대수(대)'], errors='coerce')
    
    # 컬럼명 통일
    inchun_accident_processed.columns = ['연도', '천대당발생건수', '발생건수', '등록대수']
    inchun_accident_processed['시도명'] = '인천광역시'
    inchun_accident_processed['데이터_출처'] = '인천'
    
    # 인천 데이터에는 사망자수와 부상자수 정보가 없으므로 NaN으로 설정
    inchun_accident_processed['사망자수'] = np.nan
    inchun_accident_processed['부상자수'] = np.nan
    
    # 5. 데이터 통합
    print("데이터 통합 중...")
    
    # 도로 데이터 통합
    combined_road = pd.concat([busan_road_selected, inchun_road_selected], ignore_index=True)
    
    # 교통사고 데이터 통합 (부산 데이터에 맞춰 컬럼 통일)
    busan_accident_final = busan_accident_processed[['구분', '발생건수', '사망자 수', '부상자 수', '시도명', '데이터_출처']].copy()
    busan_accident_final.columns = ['연도', '발생건수', '사망자수', '부상자수', '시도명', '데이터_출처']
    
    # 인천 데이터에서 필요한 컬럼만 선택
    inchun_accident_final = inchun_accident_processed[['연도', '발생건수', '사망자수', '부상자수', '시도명', '데이터_출처']].copy()
    
    combined_accident = pd.concat([busan_accident_final, inchun_accident_final], ignore_index=True)
    
    # 6. 데이터 정리 및 검증
    print("데이터 정리 및 검증 중...")
    
    # 결측값 확인
    print("\n=== 결측값 현황 ===")
    print("도로 데이터:")
    print(combined_road.isnull().sum())
    print("\n교통사고 데이터:")
    print(combined_accident.isnull().sum())
    
    # 기본 통계 정보
    print("\n=== 도로 데이터 기본 통계 ===")
    print(combined_road.describe())
    
    print("\n=== 교통사고 데이터 기본 통계 ===")
    print(combined_accident.describe())
    
    # 7. 데이터 저장
    print("전처리된 데이터 저장 중...")
    combined_road.to_csv("processed_road_data.csv", index=False, encoding="utf-8-sig")
    combined_accident.to_csv("processed_accident_data.csv", index=False, encoding="utf-8-sig")
    
    print("전처리 완료!")
    print(f"도로 데이터: {combined_road.shape}")
    print(f"교통사고 데이터: {combined_accident.shape}")
    
    return combined_road, combined_accident

def create_summary_statistics(road_data, accident_data):
    """
    전처리된 데이터의 요약 통계를 생성합니다.
    """
    print("\n=== 데이터 요약 통계 ===")
    
    # 시도별 도로 현황
    print("\n1. 시도별 도로 현황:")
    road_summary = road_data.groupby('시도명').agg({
        '도로명': 'count',
        '연장': ['mean', 'sum'],
        '폭': ['mean', 'sum'],
        '면적': ['mean', 'sum']
    }).round(2)
    print(road_summary)
    
    # 시도별 교통사고 현황
    print("\n2. 시도별 교통사고 현황:")
    accident_summary = accident_data.groupby('시도명').agg({
        '발생건수': ['mean', 'sum'],
        '사망자수': ['mean', 'sum'],
        '부상자수': ['mean', 'sum']
    }).round(2)
    print(accident_summary)
    
    # 연도별 교통사고 추이
    print("\n3. 연도별 교통사고 추이:")
    yearly_accident = accident_data.groupby(['연도', '시도명']).agg({
        '발생건수': 'sum',
        '사망자수': 'sum',
        '부상자수': 'sum'
    }).reset_index()
    print(yearly_accident.sort_values(['연도', '시도명']))

def visualize_outliers(road_data, accident_data):
    """
    데이터의 이상치를 시각화합니다.
    """
    print("\n=== 이상치 시각화 ===")
    
    # 1. 도로 데이터 이상치 시각화
    print("1. 도로 데이터 이상치 분석")
    
    # 숫자형 컬럼만 선택
    numeric_road_cols = ['연장', '폭', '면적']
    
    # 박스플롯으로 이상치 확인
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('도로 데이터 이상치 분석 (박스플롯)', fontsize=16)
    
    for i, col in enumerate(numeric_road_cols):
        # 시도별로 분리해서 박스플롯
        data_to_plot = [road_data[road_data['시도명'] == '부산광역시'][col].dropna(),
                       road_data[road_data['시도명'] == '인천광역시'][col].dropna()]
        
        axes[i].boxplot(data_to_plot, labels=['부산', '인천'])
        axes[i].set_title(f'{col} 분포')
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 히스토그램으로 분포 확인
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('도로 데이터 분포 분석 (히스토그램)', fontsize=16)
    
    for i, col in enumerate(numeric_road_cols):
        # 부산 데이터
        axes[0, i].hist(road_data[road_data['시도명'] == '부산광역시'][col].dropna(), 
                       bins=50, alpha=0.7, label='부산', color='blue')
        axes[0, i].set_title(f'부산 - {col} 분포')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('빈도')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # 인천 데이터
        axes[1, i].hist(road_data[road_data['시도명'] == '인천광역시'][col].dropna(), 
                       bins=50, alpha=0.7, label='인천', color='red')
        axes[1, i].set_title(f'인천 - {col} 분포')
        axes[1, i].set_xlabel(col)
        axes[1, i].set_ylabel('빈도')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


    # 2. 교통사고 데이터 이상치 시각화
    print("2. 교통사고 데이터 이상치 분석")
    
    # 숫자형 컬럼만 선택 (NaN이 아닌 컬럼)
    numeric_accident_cols = ['발생건수', '사망자수', '부상자수']
    available_cols = [col for col in numeric_accident_cols if col in accident_data.columns]
    
    # 박스플롯으로 이상치 확인
    fig, axes = plt.subplots(1, len(available_cols), figsize=(6*len(available_cols), 6))
    if len(available_cols) == 1:
        axes = [axes]
    
    fig.suptitle('교통사고 데이터 이상치 분석 (박스플롯)', fontsize=16)
    
    for i, col in enumerate(available_cols):
        # 시도별로 분리해서 박스플롯
        data_to_plot = [accident_data[accident_data['시도명'] == '부산광역시'][col].dropna(),
                       accident_data[accident_data['시도명'] == '인천광역시'][col].dropna()]
        
        axes[i].boxplot(data_to_plot, labels=['부산', '인천'])
        axes[i].set_title(f'{col} 분포')
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. 이상치 통계 정보 출력
    print("\n3. 이상치 통계 정보")
    
    for col in numeric_road_cols:
        print(f"\n{col} 이상치 분석:")
        for city in ['부산광역시', '인천광역시']:
            data = road_data[road_data['시도명'] == city][col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            print(f"  {city}:")
            print(f"    전체 데이터: {len(data)}개")
            print(f"    이상치 개수: {len(outliers)}개")
            print(f"    이상치 비율: {len(outliers)/len(data)*100:.2f}%")
            print(f"    최소값: {data.min():.2f}")
            print(f"    최대값: {data.max():.2f}")
            print(f"    이상치 하한: {lower_bound:.2f}")
            print(f"    이상치 상한: {upper_bound:.2f}")

if __name__ == "__main__":
    # 데이터 전처리 실행
    road_data, accident_data = load_and_preprocess_data()
    
    # 요약 통계 생성
    create_summary_statistics(road_data, accident_data)
    
    # 이상치 시각화
    visualize_outliers(road_data, accident_data)
    
    print("\n전처리 작업이 완료되었습니다!")
    print("생성된 파일:")
    print("- processed_road_data.csv")
    print("- processed_accident_data.csv")

# 전처리된 데이터 불러오기
road_df = pd.read_csv('data/processed_road_data.csv')
accident_df = pd.read_csv('data/processed_accident_data.csv')

# 부산 도로 데이터
busan_road_df = road_df[road_df['시도명'] == '부산광역시'].reset_index(drop=True)
# 인천 도로 데이터
inchun_road_df = road_df[road_df['시도명'] == '인천광역시'].reset_index(drop=True)

# 부산 사고 데이터
busan_accident_df = accident_df[accident_df['시도명'] == '부산광역시'].reset_index(drop=True)
# 인천 사고 데이터
inchun_accident_df = accident_df[accident_df['시도명'] == '인천광역시'].reset_index(drop=True)

# 데이터프레임 확인
print('부산 도로 데이터프레임 shape:', busan_road_df.shape)
print('인천 도로 데이터프레임 shape:', inchun_road_df.shape)
print('부산 사고 데이터프레임 shape:', busan_accident_df.shape)
print('인천 사고 데이터프레임 shape:', inchun_accident_df.shape)

busan_road_df.head() 