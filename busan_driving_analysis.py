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

# 데이터 존재 여부 확인
print(f"부산 도로 데이터: {len(busan_road_df)}건")
print(f"인천 도로 데이터: {len(inchun_road_df)}건")
print(f"부산 교통사고 데이터: {len(busan_accident_df)}건")
print(f"인천 교통사고 데이터: {len(inchun_accident_df)}건")

# 인천 데이터가 없는 경우 경고
if len(inchun_road_df) == 0:
    print("\n⚠️ 경고: 인천 도로 데이터가 없습니다. 부산 데이터만으로 분석을 진행합니다.")
    comparison_available = False
else:
    comparison_available = True

if len(inchun_accident_df) == 0:
    print("⚠️ 경고: 인천 교통사고 데이터가 없습니다.")
    accident_comparison_available = False
else:
    accident_comparison_available = True

print("=== 부산 운전 난이도 분석 ===")

# 1. 도로 인프라 분석
print("\n1. 도로 인프라 분석")

# 도로 위계별 분석
print("1-1. 도로 위계별 현황")
road_hierarchy_busan = busan_road_df['도로위계'].value_counts()
road_hierarchy_inchun = inchun_road_df['도로위계'].value_counts()

print("부산 도로 위계:")
print(road_hierarchy_busan)
print("\n인천 도로 위계:")
print(road_hierarchy_inchun)

# 도로 폭 분석 (좁은 도로가 많을수록 운전이 어려움)
print("\n1-2. 도로 폭 분석")
narrow_roads_busan = busan_road_df[busan_road_df['폭'] <= 6].shape[0]  # 6m 이하를 좁은 도로로 정의
narrow_roads_inchun = inchun_road_df[inchun_road_df['폭'] <= 6].shape[0]

print(f"부산 좁은 도로(6m 이하) 비율: {narrow_roads_busan}/{len(busan_road_df)} ({narrow_roads_busan/len(busan_road_df)*100:.1f}%)")
print(f"인천 좁은 도로(6m 이하) 비율: {narrow_roads_inchun}/{len(inchun_road_df)} ({narrow_roads_inchun/len(inchun_road_df)*100:.1f}%)")

# 평균 도로 폭 비교
avg_width_busan = busan_road_df['폭'].mean()
avg_width_inchun = inchun_road_df['폭'].mean()
print(f"부산 평균 도로 폭: {avg_width_busan:.1f}m")
print(f"인천 평균 도로 폭: {avg_width_inchun:.1f}m")

# 2. 교통사고 분석
print("\n2. 교통사고 분석")

# 연도별 교통사고 추이
print("2-1. 연도별 교통사고 발생건수")
yearly_accidents = busan_accident_df.groupby('연도')['발생건수'].sum().reset_index()
print(yearly_accidents)

# 사망자수와 부상자수 분석
print("\n2-2. 사망자수 및 부상자수 분석")
if '사망자수' in busan_accident_df.columns:
    total_deaths = busan_accident_df['사망자수'].sum()
    total_injuries = busan_accident_df['부상자수'].sum()
    print(f"부산 총 사망자수: {total_deaths}")
    print(f"부산 총 부상자수: {total_injuries}")
    print(f"부산 평균 연간 사망자수: {total_deaths/len(busan_accident_df):.1f}")
    print(f"부산 평균 연간 부상자수: {total_injuries/len(busan_accident_df):.1f}")

# 3. 도로 복잡도 분석
print("\n3. 도로 복잡도 분석")

# 시군구별 도로 밀도
print("3-1. 시군구별 도로 현황")
district_roads = busan_road_df['시군구명'].value_counts()
print("부산 시군구별 도로 개수:")
print(district_roads.head(10))

# 도로 위계별 평균 폭
print("\n3-2. 도로 위계별 평균 폭")
hierarchy_width = busan_road_df.groupby('도로위계')['폭'].agg(['mean', 'count']).round(2)
print(hierarchy_width)

# 4. 운전 난이도 지표 계산
print("\n4. 운전 난이도 지표 계산")

# 지표 1: 좁은 도로 비율 (높을수록 어려움)
narrow_road_ratio_busan = narrow_roads_busan / len(busan_road_df)
if comparison_available:
    narrow_road_ratio_inchun = narrow_roads_inchun / len(inchun_road_df)

# 지표 2: 평균 도로 폭 (낮을수록 어려움)
width_score_busan = 1 - (avg_width_busan / 60)  # 20m를 최대값으로 가정
if comparison_available:
    width_score_inchun = 1 - (avg_width_inchun / 60)

# 지표 3: 교통사고 발생률 (높을수록 어려움)
accident_rate_busan = busan_accident_df['발생건수'].mean() / 1000  # 천 건당
if accident_comparison_available:
    accident_rate_inchun = inchun_accident_df['발생건수'].mean() / 1000

# 종합 난이도 점수 (0-1, 높을수록 어려움)
difficulty_score_busan = (narrow_road_ratio_busan * 0.4 + 
                         width_score_busan * 0.3 + 
                         min(accident_rate_busan, 1) * 0.3)

if comparison_available and accident_comparison_available:
    difficulty_score_inchun = (narrow_road_ratio_inchun * 0.4 + 
                              width_score_inchun * 0.3 + 
                              min(accident_rate_inchun, 1) * 0.3)

print(f"부산 운전 난이도 점수: {difficulty_score_busan:.3f}")
if comparison_available and accident_comparison_available:
    print(f"인천 운전 난이도 점수: {difficulty_score_inchun:.3f}")

# 5. 시각화
print("\n5. 시각화")

# 도로 폭 분포 비교
if comparison_available:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('부산 vs 인천 도로 폭 분포 비교', fontsize=16)

    axes[0].hist(busan_road_df['폭'].dropna(), bins=30, alpha=0.7, label='부산', color='blue')
    axes[0].set_title('부산 도로 폭 분포')
    axes[0].set_xlabel('도로 폭 (m)')
    axes[0].set_ylabel('빈도')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(inchun_road_df['폭'].dropna(), bins=30, alpha=0.7, label='인천', color='red')
    axes[1].set_title('인천 도로 폭 분포')
    axes[1].set_xlabel('도로 폭 (m)')
    axes[1].set_ylabel('빈도')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 도로 위계별 분포
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('도로 위계별 분포', fontsize=16)

    road_hierarchy_busan.plot(kind='bar', ax=axes[0], color='blue', alpha=0.7)
    axes[0].set_title('부산 도로 위계별 분포')
    axes[0].set_xlabel('도로 위계')
    axes[0].set_ylabel('도로 개수')
    axes[0].tick_params(axis='x', rotation=45)

    if comparison_available:
        road_hierarchy_inchun = inchun_road_df['도로위계'].value_counts()
        if len(road_hierarchy_inchun) > 0:
            road_hierarchy_inchun.plot(kind='bar', ax=axes[1], color='red', alpha=0.7)
            axes[1].set_title('인천 도로 위계별 분포')
            axes[1].set_xlabel('도로 위계')
            axes[1].set_ylabel('도로 개수')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, '인천 도로위계 데이터 없음', ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('인천 도로 위계별 분포')
    else:
        axes[1].text(0.5, 0.5, '인천 데이터 없음', ha='center', va='center', transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('인천 도로 위계별 분포')

    plt.tight_layout()
    plt.show()

    # 운전 난이도 점수 비교
    if comparison_available and accident_comparison_available:
        fig, ax = plt.subplots(figsize=(10, 6))
        cities = ['부산', '인천']
        scores = [difficulty_score_busan, difficulty_score_inchun]
        colors = ['blue', 'red']
        
        bars = ax.bar(cities, scores, color=colors, alpha=0.7)
        ax.set_title('부산 vs 인천 운전 난이도 점수 비교', fontsize=14)
        ax.set_ylabel('난이도 점수 (높을수록 어려움)')
        ax.set_ylim(0, 1)
        
        # 막대 위에 점수 표시
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.show()

else:
    # 부산 데이터만으로 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('부산 도로 데이터 분석', fontsize=16)

    # 도로 폭 분포
    axes[0].hist(busan_road_df['폭'].dropna(), bins=30, alpha=0.7, color='blue')
    axes[0].set_title('부산 도로 폭 분포')
    axes[0].set_xlabel('도로 폭 (m)')
    axes[0].set_ylabel('빈도')
    axes[0].grid(True, alpha=0.3)

    # 도로 위계별 분포
    road_hierarchy_busan.plot(kind='bar', ax=axes[1], color='blue', alpha=0.7)
    axes[1].set_title('부산 도로 위계별 분포')
    axes[1].set_xlabel('도로 위계')
    axes[1].set_ylabel('도로 개수')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # 부산 운전 난이도 점수
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['부산'], [difficulty_score_busan], color='blue', alpha=0.7)
    ax.set_title('부산 운전 난이도 점수', fontsize=14)
    ax.set_ylabel('난이도 점수 (높을수록 어려움)')
    ax.set_ylim(0, 1)
    
    # 점수 표시
    ax.text(0, difficulty_score_busan + 0.01, f'{difficulty_score_busan:.3f}', 
            ha='center', va='bottom', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.show()

# 6. 결론 도출
print("\n6. 결론")

print("=== 부산 운전 난이도 분석 결과 ===")
print(f"1. 좁은 도로(6m 이하) 비율: {narrow_road_ratio_busan*100:.1f}%")
print(f"2. 평균 도로 폭: {avg_width_busan:.1f}m")
print(f"3. 연간 평균 교통사고 발생건수: {busan_accident_df['발생건수'].mean():.0f}건")

if difficulty_score_busan > 0.6:
    difficulty_level = "매우 어려움"
elif difficulty_score_busan > 0.4:
    difficulty_level = "어려움"
elif difficulty_score_busan > 0.2:
    difficulty_level = "보통"
else:
    difficulty_level = "쉬움"

print(f"\n종합 운전 난이도: {difficulty_level} (점수: {difficulty_score_busan:.3f})")

# 주요 어려운 점들
print("\n=== 부산 운전의 주요 어려운 점 ===")
if narrow_road_ratio_busan > 0.5:
    print("• 좁은 도로가 많아 차량 통행이 어려움")
if avg_width_busan < 8:
    print("• 평균 도로 폭이 좁아 주차 및 회전이 어려움")
if busan_accident_df['발생건수'].mean() > 12000:
    print("• 교통사고 발생률이 높음")

# 부산의 지형적 특성 고려
print("\n=== 부산의 지형적 특성 ===")
print("• 산지와 해안이 복합된 복잡한 지형")
print("• 급경사 도로와 터널이 많음")
print("• 좁은 골목길과 계단식 도로 구조")
print("• 해안가와 산지의 급커브 구간")

print("\n=== 최종 결론 ===")
if difficulty_score_busan > 0.5:
    print("부산은 상대적으로 운전하기 어려운 도시입니다.")
    print("주요 이유:")
    print("- 좁은 도로가 많음")
    print("- 복잡한 지형 구조")
    print("- 높은 교통사고 발생률")
else:
    print("부산은 운전하기 보통 수준의 도시입니다.")

# 인천과의 비교
if comparison_available and difficulty_score_busan > difficulty_score_inchun:
    print(f"\n인천({difficulty_score_inchun:.3f}) 대비 부산({difficulty_score_busan:.3f})이 운전하기 더 어려운 도시입니다.")
elif comparison_available and difficulty_score_busan < difficulty_score_inchun:
    print(f"\n인천({difficulty_score_inchun:.3f}) 대비 부산({difficulty_score_busan:.3f})이 운전하기 더 쉬운 도시입니다.")
else:
    print("\n인천과의 비교 불가") 