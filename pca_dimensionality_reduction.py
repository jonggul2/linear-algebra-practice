# %% [markdown]
"""
# Principal Component Analysis for Dimensionality Reduction

차원 축소(Dimensionality Reduction)는 데이터의 본질적인 구조는 최대한 유지하면서, 데이터를 표현하는 변수(차원)의 수를 줄이는 기술입니다.
고차원 데이터에서 불필요한 노이즈를 제거하고, 시각화를 용이하게 하며, 머신러닝 모델의 학습 속도를 높이는 등 다양한 이점을 가집니다.

**주성분 분석(Principal Component Analysis, PCA)**은 가장 널리 사용되는 차원 축소 기법 중 하나로, 데이터의 분산(variance)이 가장 큰 방향을 새로운 좌표축(주성분)으로 설정하여 데이터를 선형 변환하는 방식입니다.

본 실습에서는 다음 두 가지를 학습합니다.
1.  **PCA 알고리즘의 단계별 구현**: 2차원 예제 데이터를 통해 PCA의 각 단계를 직접 구현하며 원리를 이해합니다.
2.  **고차원 데이터 적용**: MNIST 손글씨 이미지 데이터에 PCA를 적용하여 차원 축소의 효과와 응용을 확인합니다.
"""

# %% [markdown]
"""
### 준비하기: 라이브러리 설치 및 불러오기

본 실습에서는 데이터 분석과 과학 계산의 필수 도구인 `numpy`와, 데이터 시각화를 위한 `matplotlib` 라이브러리를 사용합니다.
아래 코드는 라이브러리들이 설치되어 있는지 확인하고, 만약 없다면 자동으로 설치합니다.
"""

# %%
# --- 라이브러리 설치 ---
import subprocess
import sys

def install_if_not_exists(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_if_not_exists("numpy")
install_if_not_exists("matplotlib")


# --- 라이브러리 불러오기 ---
import numpy as np
import matplotlib.pyplot as plt

# 재현성을 위해 랜덤 시드 고정
np.random.seed(0)


# %% [markdown]
"""
---
## Part 1: PCA 알고리즘 단계별 구현 (2D 예제)
---
단순한 2차원 데이터를 통해 PCA가 작동하는 각 단계를 명확하게 이해하는 데 초점을 맞춥니다.
"""

# %% [markdown]
"""
### 1.1. 데이터 생성 및 시각화
두 변수 간에 강한 양의 상관관계가 있는 2차원 데이터셋을 생성하고, 산점도(scatter plot)를 통해 데이터의 분포 방향을 확인합니다.
"""

# %%
# 실습: x와 y의 관계(예: y = -2 * x)나 노이즈의 크기를 바꿔보며 데이터 분포가 어떻게 변하는지 확인해보세요.
num_samples = 100
# 원점에서 의도적으로 벗어난 데이터를 생성하여, 평균 중심화의 효과를 명확히 보여줍니다.
x = np.linspace(3, 5, num_samples)
y = 2 * x - 4 + np.random.normal(0, 0.5, num_samples)
data_2d = np.array([x, y]).T

plt.figure(figsize=(8, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.7)
plt.title("Original 2D Data (Not Centered)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis('equal')
plt.grid(True)
plt.show()

# %% [markdown]
"""
### 1.2. 데이터 전처리: 평균 중심화 (Mean Centering)
PCA는 데이터의 분산을 기반으로 하므로, 각 변수(feature)의 평균을 0으로 맞추는 것이 필수적입니다.
데이터의 각 축(x, y)에 대해 평균을 계산하고, 모든 데이터 포인트에서 해당 평균을 빼줍니다.
"""

# %%
mean_vec = np.mean(data_2d, axis=0)
centered_data = data_2d - mean_vec

plt.figure(figsize=(8, 6))
plt.scatter(centered_data[:, 0], centered_data[:, 1], alpha=0.7, color='orange')
plt.title("Centered Data (Mean = 0)")
plt.xlabel("Feature 1 (centered)")
plt.ylabel("Feature 2 (centered)")
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.axis('equal')
plt.grid(True)
plt.show()

# %% [markdown]
"""
### 1.3. 공분산 행렬 계산 및 고유값 분해
PCA의 핵심은 데이터의 분산이 가장 큰 방향(주성분)을 찾는 것이며, 이는 **공분산 행렬의 고유벡터(eigenvector)**에 해당합니다.
- **공분산 행렬**: 데이터가 각 축을 따라 얼마나 퍼져 있고, 축 간에는 어떤 상관관계를 갖는지 나타내는 행렬입니다.
- **고유값 분해**: 이 공분산 행렬을 고유값(eigenvalue)과 고유벡터로 분해합니다.
  - **고유벡터**: 데이터의 분산이 가장 큰 방향, 즉 **주성분(Principal Components)**이 됩니다.
  - **고유값**: 해당 고유벡터 방향으로 데이터가 가진 분산의 크기를 나타냅니다.
"""

# %%
# 1. 공분산 행렬 계산
#    np.cov는 (변수 개수, 샘플 수) 형태의 입력을 기대하므로 전치(.T)가 필요합니다.
cov_matrix = np.cov(centered_data.T)

# 2. 고유값 분해
#    np.linalg.eigh는 대칭 행렬에 사용되며, 고유값을 오름차순으로 정렬하여 반환합니다.
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 고유값이 큰 순서대로 정렬 (내림차순)
sort_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_indices]
eigenvectors = eigenvectors[:, sort_indices]

pc1 = eigenvectors[:, 0]
pc2 = eigenvectors[:, 1]

print("공분산 행렬:\n", np.round(cov_matrix, 2))
print("\n고유값 (분산 크기):\n", np.round(eigenvalues, 2))
print("\n고유벡터 (주성분 방향):\n", np.round(eigenvectors, 2))
print("\n제1 주성분 (PC1):", np.round(pc1, 2))
print("제2 주성분 (PC2):", np.round(pc2, 2))

# %% [markdown]
"""
### 1.4. 주성분 시각화 및 데이터 사영
계산된 두 주성분(PC1, PC2)을 데이터 산점도 위에 화살표로 그려, 데이터의 주된 분산 방향과 일치하는지 확인합니다.
그 후, 데이터를 1차원으로 축소하기 위해 중심화된 데이터를 제1 주성분(PC1) 벡터에 **사영(projection)**합니다.
"""

# %%
# 주성분 시각화
plt.figure(figsize=(8, 6))
plt.scatter(centered_data[:, 0], centered_data[:, 1], alpha=0.7, color='orange')
# 고유벡터(주성분)를 화살표로 그립니다. 길이는 해당 방향의 분산 크기(고유값의 제곱근)를 반영합니다.
plt.quiver(0, 0, pc1[0] * np.sqrt(eigenvalues[0]), pc1[1] * np.sqrt(eigenvalues[0]),
           color='red', scale_units='xy', scale=1, width=0.01, label='PC1')
plt.quiver(0, 0, pc2[0] * np.sqrt(eigenvalues[1]), pc2[1] * np.sqrt(eigenvalues[1]),
           color='blue', scale_units='xy', scale=1, width=0.01, label='PC2')
plt.title("Principal Components on Centered Data")
plt.xlabel("Feature 1 (centered)")
plt.ylabel("Feature 2 (centered)")
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# 데이터 사영 (2D -> 1D)
projected_data = centered_data @ pc1
print("원본 2D 데이터 (첫 5개):\n", np.round(data_2d[:5], 2))
print("\n중심화된 2D 데이터 (첫 5개):\n", np.round(centered_data[:5], 2))
print("\nPC1으로 사영된 1D 데이터 (첫 5개):\n", np.round(projected_data[:5], 2))

# %% [markdown]
"""
### 1.5. 데이터 재구성 (Back-projection)
1차원으로 축소된 데이터(`projected_data`)를 다시 2차원 공간으로 되돌리는 과정입니다.
이 과정은 `(1D 데이터) @ (PC1 벡터의 전치)`로 수행할 수 있으며, 결과적으로 모든 데이터 포인트가 제1 주성분(PC1) 직선 위에 놓이게 됩니다.
이는 PCA가 정보의 손실을 감수하고 데이터의 가장 중요한 분산 방향으로 데이터를 '근사'하는 과정을 시각적으로 보여줍니다.
"""

# %%
# 1. 1D 데이터를 다시 2D 공간으로 재구성 (중심화된 공간 기준)
#    1D 벡터인 projected_data를 열 벡터로, pc1을 행 벡터로 변환하여 외적(outer product)을 수행합니다.
reconstructed_centered_data = projected_data[:, np.newaxis] @ pc1[np.newaxis, :]

# 2. 원본 데이터 공간으로 이동 (평균을 다시 더해줌)
reconstructed_data = reconstructed_centered_data + mean_vec

# 재구성 결과 출력
print("--- 데이터 재구성 계산 과정 (첫 1개 포인트) ---")
projected_val = projected_data[0]
reconstructed_centered_pt = reconstructed_centered_data[0]
reconstructed_pt = reconstructed_data[0]

print(f"  - 1D로 축소된 값: {projected_val:.2f}")
print(f"  - 2D로 재구성된 중심화 좌표: {np.round(reconstructed_centered_pt, 2)}  (계산: {projected_val:.2f} * {np.round(pc1, 2)})")
print(f"  - 최종 재구성 좌표: {np.round(reconstructed_pt, 2)}  (계산: {np.round(reconstructed_centered_pt, 2)} + {np.round(mean_vec, 2)})")


# 시각화
plt.figure(figsize=(8, 8))
# 원본 데이터
plt.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.5, s=100, label='Original Data')
# 재구성된 데이터
plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], color='red', marker='x', s=100, label='Reconstructed Data (on PC1)')

# 원본-재구성 연결선 (오차 시각화)
for i in range(len(data_2d)):
    plt.plot([data_2d[i, 0], reconstructed_data[i, 0]], [data_2d[i, 1], reconstructed_data[i, 1]], 'k--', alpha=0.4)

# 주성분 직선 (평균을 고려하여 원본 데이터 공간에 표시)
line_range = np.linspace(-2.5, 2.5, 100)
line_x = line_range * pc1[0] + mean_vec[0]
line_y = line_range * pc1[1] + mean_vec[1]
plt.plot(line_x, line_y, 'r-', alpha=0.7, label='PC1 Line')

plt.title("Original vs. Reconstructed Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
"""
---
## Part 2: 실제 데이터 적용 (MNIST 숫자 이미지)
---
고차원 데이터인 MNIST 손글씨 숫자 이미지를 PCA로 분석하여 차원 축소의 강력한 응용을 경험합니다.
각 이미지는 28x28=784개의 픽셀로 이루어진 784차원 벡터입니다.
"""

# %% [markdown]
"""
### 2.1. 데이터 로딩 및 준비
전체 MNIST 데이터셋(0~9) 중 사용자가 선택한 숫자들을 사용하여 784차원의 이미지를 저차원으로 축소하고 분석합니다.
"""

# %%
# --- 하이퍼파라미터 설정 ---
# 실습: 분석하고 싶은 숫자들을 리스트에 포함시켜 보세요. 기본값은 [0, 1] 입니다.
# 예: [3, 8], [0, 1, 7], list(range(10))
selected_digits = [0, 1]


# K-means 실습에서 생성한 mnist.npz 파일 사용
with np.load('mnist.npz') as data:
    X_full = data['X']
    y_full = data['y']

# 선택된 숫자에 해당하는 데이터만 필터링합니다.
filter_mask = np.isin(y_full, selected_digits)
X_mnist = X_full[filter_mask]
y_mnist = y_full[filter_mask]

print(f"선택된 숫자: {selected_digits}")
print("선택된 데이터 크기:", X_mnist.shape)


# 데이터 중심화
X_centered = X_mnist - np.mean(X_mnist, axis=0)

# %% [markdown]
"""
### 2.2. PCA 실행 및 분산 설명량 시각화
선택된 숫자 이미지 데이터에 PCA를 적용하고, 각 주성분이 전체 데이터 분산의 몇 %를 설명하는지 시각화합니다.
**Scree Plot**은 각 주성분(고유값)의 중요도를 시각적으로 보여주며, "팔꿈치(elbow)" 지점을 통해 사용할 주성분의 수를 결정하는 데 도움을 줍니다.
누적 분산 설명량 그래프는 몇 개의 주성분을 사용해야 원하는 비율(예: 95%)의 데이터 정보를 보존할 수 있는지 알려줍니다.
"""

# %%
# 공분산 행렬 및 고유값 분해
cov_matrix_mnist = np.cov(X_centered.T)
eigenvalues_mnist, eigenvectors_mnist = np.linalg.eigh(cov_matrix_mnist)

# 고유값/고유벡터 정렬 (내림차순)
sort_indices_mnist = np.argsort(eigenvalues_mnist)[::-1]
eigenvalues_mnist = eigenvalues_mnist[sort_indices_mnist]
eigenvectors_mnist = eigenvectors_mnist[:, sort_indices_mnist]

# 분산 설명량 계산
explained_variance_ratio = eigenvalues_mnist / np.sum(eigenvalues_mnist)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# 시각화할 주성분 개수를 데이터 차원 수와 20 중 작은 값으로 제한
num_components_to_plot = min(20, X_centered.shape[1])
plt.bar(range(1, num_components_to_plot + 1), explained_variance_ratio[:num_components_to_plot], alpha=0.8, label='Individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.title('Scree Plot (Top Components on Selected Digits)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='.', linestyle='--')
plt.axhline(y=0.95, color='r', linestyle='-', label='95% threshold')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Cumulative Explained Variance on Selected Digits')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 95% 분산을 설명하는 데 필요한 주성분 개수 확인
# np.argmax는 조건이 처음 True가 되는 인덱스를 반환합니다.
# 누적 분산이 없는 경우(데이터가 1개뿐인 등)를 대비하여 기본값 설정
if len(cumulative_explained_variance) > 0:
    n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
    print(f"데이터 분산의 95%를 설명하는 데 필요한 주성분 개수: {n_components_95}")
else:
    print("분산 설명량을 계산할 수 없습니다.")


# %% [markdown]
"""
### 2.3. 저차원 임베딩 시각화
784차원의 이미지 데이터를 **가장 중요한 두 개의 주성분(PC1, PC2)**만 사용하여 2차원으로 축소하고, 그 결과를 산점도로 시각화합니다.
PCA는 레이블 정보 없이(Unsupervised) 오직 데이터의 분산만을 기반으로 동작했음에도, 서로 다른 숫자들을 어느 정도 군집화하는 경향을 보여줍니다.
"""

# %%
# PC1, PC2 추출
pc1_mnist = eigenvectors_mnist[:, 0]
pc2_mnist = eigenvectors_mnist[:, 1]

# 2차원으로 사영
projected_mnist = np.c_[X_centered @ pc1_mnist, X_centered @ pc2_mnist]

# 시각화
plt.figure(figsize=(12, 10))
# cmap='tab10'은 10개의 클래스를 구분하기 좋은 컬러맵입니다.
scatter = plt.scatter(projected_mnist[:, 0], projected_mnist[:, 1], c=y_mnist, cmap=plt.get_cmap("tab10", 10), alpha=0.6, s=10)
plt.title("Selected MNIST Digits Projected onto First Two Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
# 컬러바를 추가하여 각 색상이 어떤 숫자를 나타내는지 표시합니다.
# 선택된 숫자에 대해서만 눈금을 표시합니다.
cbar = plt.colorbar(scatter, ticks=selected_digits)
cbar.set_label('Digit')
plt.grid(True)
plt.show()

# %% [markdown]
"""
### 2.4. 이미지 재구성을 통한 차원 축소 확인
주성분의 개수(`k`)를 다르게 하여 원본 이미지를 재구성해봅니다.
적은 수의 주성분만으로도 원본 이미지의 특징이 대부분 복원되는 것을 통해, PCA가 데이터의 핵심 정보를 소수의 차원에 압축하는 효과적인 도구임을 이해할 수 있습니다.
"""

# %%
# 실습: k_values 리스트의 값을 바꿔보며(예: 200, 300 추가), 몇 개의 주성분으로 이미지가 거의 완벽하게 복원되는지 확인해보세요.
def reconstruct_image(data, eigenvectors, k):
    # k개의 주성분 선택
    top_k_pcs = eigenvectors[:, :k]
    # 데이터를 k차원으로 사영 (압축)
    projected_data = data @ top_k_pcs
    # k차원 데이터로부터 원본 차원으로 재구성 (압축 해제)
    reconstructed_data = projected_data @ top_k_pcs.T
    return reconstructed_data

# 원본 이미지와 비교할 다양한 숫자 샘플 선택
# 선택된 각 숫자별로 첫 번째 이미지 인덱스를 찾습니다.
sample_indices = []
if len(X_mnist) > 0:
    for digit in selected_digits:
        # np.where는 튜플을 반환하므로 [0][0]으로 인덱스를 추출합니다.
        # 해당 숫자가 데이터에 없을 경우를 대비하여 예외 처리를 합니다.
        try:
            sample_indices.append(np.where(y_mnist == digit)[0][0])
        except IndexError:
            print(f"경고: 데이터셋에 숫자 {digit}이(가) 없어 재구성 예시에서 제외됩니다.")

    sample_images = X_centered[sample_indices]
    original_images_to_show = X_mnist[sample_indices]

    k_values = [1, 10, 50, 100, 300]
    num_k = len(k_values)
    num_samples_to_show = len(sample_indices)

    if num_samples_to_show > 0:
        fig, axes = plt.subplots(num_samples_to_show, num_k + 1, figsize=(num_k * 2, num_samples_to_show * 2))

        # 단일 샘플일 경우 axes가 1D 배열이 되므로 2D로 만듭니다.
        if num_samples_to_show == 1:
            axes = axes.reshape(1, -1)

        for i, img_idx in enumerate(sample_indices):
            # 원본 이미지 표시
            axes[i, 0].imshow(original_images_to_show[i].reshape(28, 28), cmap='gray')
            axes[i, 0].set_title(f"Original ({y_mnist[img_idx]})")
            axes[i, 0].axis('off')

            # 재구성된 이미지 표시
            for j, k in enumerate(k_values):
                # k가 데이터 차원보다 크면 안됩니다.
                if k > X_centered.shape[1]:
                    axes[i, j + 1].set_title(f"k={k} (N/A)")
                    axes[i, j + 1].axis('off')
                    continue
                    
                reconstructed = reconstruct_image(X_centered, eigenvectors_mnist, k)
                reconstructed_img = reconstructed[img_idx] + np.mean(X_mnist, axis=0)
                axes[i, j + 1].imshow(reconstructed_img.reshape(28, 28), cmap='gray')
                axes[i, j + 1].set_title(f"k={k}")
                axes[i, j + 1].axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("재구성할 이미지를 찾을 수 없습니다.")
else:
    print("데이터가 없어 이미지 재구성을 건너뜁니다.") 