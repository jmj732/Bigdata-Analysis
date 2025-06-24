import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS용
plt.rcParams['axes.unicode_minus'] = False

# 전처리된 데이터 불러오기
road_df = pd.read_csv('data/processed_road_data.csv')
accident_df = pd.read_csv('data/processed_accident_data.csv')

# 부산/인천 데이터 분리
busan_road_df = road_df[road_df['시도명'] == '부산광역시'].reset_index(drop=True)
inchun_road_df = road_df[road_df['시도명'] == '인천광역시'].reset_index(drop=True)
busan_accident_df = accident_df[accident_df['시도명'] == '부산광역시'].reset_index(drop=True)
inchun_accident_df = accident_df[accident_df['시도명'] == '인천광역시'].reset_index(drop=True)

print("=== 이상치 시각화 ===")

# 1. 도로 데이터 이상치 시각화
print("1. 도로 데이터 이상치 분석")

# 숫자형 컬럼만 선택
numeric_road_cols = ['연장', '폭', '면적']

# 박스플롯으로 이상치 확인
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('도로 데이터 이상치 분석 (박스플롯)', fontsize=16)

for i, col in enumerate(numeric_road_cols):
    # 시도별로 분리해서 박스플롯
    data_to_plot = [busan_road_df[col].dropna(),
                   inchun_road_df[col].dropna()]
    
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
    axes[0, i].hist(busan_road_df[col].dropna(), 
                   bins=50, alpha=0.7, label='부산', color='blue')
    axes[0, i].set_title(f'부산 - {col} 분포')
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel('빈도')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # 인천 데이터
    axes[1, i].hist(inchun_road_df[col].dropna(), 
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
available_cols = [col for col in numeric_accident_cols if col in accident_df.columns]

# 박스플롯으로 이상치 확인
fig, axes = plt.subplots(1, len(available_cols), figsize=(6*len(available_cols), 6))
if len(available_cols) == 1:
    axes = [axes]

fig.suptitle('교통사고 데이터 이상치 분석 (박스플롯)', fontsize=16)

for i, col in enumerate(available_cols):
    # 시도별로 분리해서 박스플롯
    data_to_plot = [busan_accident_df[col].dropna(),
                   inchun_accident_df[col].dropna()]
    
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
    for city_df, city_name in [(busan_road_df, '부산'), (inchun_road_df, '인천')]:
        data = city_df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        print(f"  {city_name}:")
        print(f"    전체 데이터: {len(data)}개")
        print(f"    이상치 개수: {len(outliers)}개")
        print(f"    이상치 비율: {len(outliers)/len(data)*100:.2f}%")
        print(f"    최소값: {data.min():.2f}")
        print(f"    최대값: {data.max():.2f}")
        print(f"    이상치 하한: {lower_bound:.2f}")
        print(f"    이상치 상한: {upper_bound:.2f}")

# 4. 데이터프레임 형태 출력
print("\n4. 데이터프레임 형태")
print('부산 도로 데이터프레임 shape:', busan_road_df.shape)
print('인천 도로 데이터프레임 shape:', inchun_road_df.shape)
print('부산 사고 데이터프레임 shape:', busan_accident_df.shape)
print('인천 사고 데이터프레임 shape:', inchun_accident_df.shape)

# 5. 각 데이터프레임 미리보기
print("\n5. 부산 도로 데이터 미리보기")
print(busan_road_df.head())

print("\n6. 인천 도로 데이터 미리보기")
print(inchun_road_df.head())

print("\n7. 부산 사고 데이터 미리보기")
print(busan_accident_df.head())

print("\n8. 인천 사고 데이터 미리보기")
print(inchun_accident_df.head()) 