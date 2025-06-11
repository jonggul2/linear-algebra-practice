# %% [markdown]
"""
# Vector Manipulation Practice

**목표**: 벡터 개념을 Numpy를 이용해 직접 구현하고, 기본 연산 및 측정 방법을 익힙니다.
"""

# %% [markdown]
"""
### 라이브러리 설치 및 불러오기
아래 코드는 실습에 필요한 `numpy`와 `matplotlib` 라이브러리가 설치되어 있는지 확인하고, 없을 경우 설치합니다.
그 다음, 실습에 필요한 모든 라이브러리를 불러옵니다.
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

# %% [markdown]
"""
### 벡터 생성 및 기본 확인
"""

# %% [markdown]
"""
#### 기본 벡터 생성하기

`np.array()`를 사용해 예시 벡터를 만들어봅니다.
"""

# %%
# 간단한 정수 벡터를 생성합니다.
a = np.array([1, 2, 3])
# 실수를 포함한 벡터를 생성할 수도 있습니다.
b = np.array([10, 20, 30, 40])
print("벡터 a:", a)
print("벡터 b:", b)

# %% [markdown]
"""
#### 벡터 속성 확인하기

벡터의 크기(`.shape`)와 요소의 개수(`.size`)를 확인합니다.
"""

# %%
print("a의 형태(크기):", a.shape)
print("a의 요소 개수:", a.size)
print("---")
print("b의 형태(크기):", b.shape)
print("b의 요소 개수:", b.size)

# %% [markdown]
"""
#### 특정 요소에 접근하기 (Indexing)

Python의 인덱스는 0부터 시작합니다.
"""

# %%
# a 벡터의 인덱스 1에 접근합니다.
print(f"a[1]: {a[1]}")

# b 벡터의 인덱스 2에 접근합니다.
print(f"b[2]: {b[2]}")

# %% [markdown]
"""
#### 특별한 벡터 만들기
`np.zeros()`와 `np.ones()`를 이용해 영 벡터와 1 벡터를 생성합니다.
"""

# %%
# 모든 요소가 0인 벡터를 생성합니다.
zero_vec = np.zeros(4)
print("영 벡터 (zeros):\n", zero_vec)
print("-" * 20)

# 모든 요소가 1인 벡터를 생성합니다.
ones_vec = np.ones(3)
print("1 벡터 (ones):\n", ones_vec)

# %% [markdown]
"""
### 벡터 덧셈, 뺄셈, 스칼라 곱
"""

# %% [markdown]
"""
#### 덧셈과 뺄셈
크기가 같은 두 벡터를 만들고 덧셈/뺄셈 연산을 수행합니다. Numpy가 요소별(element-wise)로 알아서 계산해준다는 점을 보여줍니다.
"""

# %%
# 시각화를 위해 2차원 벡터를 사용합니다.
# 시각적으로 구분이 쉽도록 서로 다른 방향의 벡터를 사용합니다.
v1 = np.array([2, 4])
v2 = np.array([4, 1])

# 덧셈
v_add = v1 + v2
print(f"v1 + v2 = {v1} + {v2} = {v_add}")

# 뺄셈
v_sub = v1 - v2
print(f"v1 - v2 = {v1} - {v2} = {v_sub}")

# %% [markdown]
"""
#### 덧셈 시각화하기
`matplotlib` 라이브러리를 사용하여 벡터 덧셈을 2D 그래프에 시각화합니다.
- **덧셈**: `v1` 벡터의 끝에서 `v2` 벡터를 이어 붙인 결과는 원점에서 시작한 `v1 + v2` 벡터와 같습니다 (평행사변형 법칙).
"""

# %%
# 원점 (Origin)
origin = np.array([0, 0])

# 그래프에 표시될 모든 점들을 모아 축의 범위를 동적으로 설정합니다.
all_points_x = [0, v1[0], v_add[0]]
all_points_y = [0, v1[1], v_add[1]]

plt.figure()
plt.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='r', label=f'v1 = {v1}')
# v2를 v1의 끝점에서 평행이동하여 표시합니다.
plt.quiver(*v1, *v2, angles='xy', scale_units='xy', scale=1, color='b', label=f'v2 (translated) = {v2}')
plt.quiver(*origin, *v_add, angles='xy', scale_units='xy', scale=1, color='g', label=f'v1 + v2 = {v_add}')

# 동적으로 계산된 범위에 여백을 주어 x, y축 범위를 설정합니다.
plt.xlim(min(all_points_x) - 1, max(all_points_x) + 1)
plt.ylim(min(all_points_y) - 1, max(all_points_y) + 1)

plt.title('Vector Addition')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid()
plt.legend()
plt.gca().set_aspect('equal', adjustable='box') # x, y축의 비율을 동일하게 설정합니다.
plt.show()


# %% [markdown]
"""
#### 뺄셈 시각화하기
`matplotlib` 라이브러리를 사용하여 벡터 뺄셈을 2D 그래프에 시각화합니다.
- **뺄셈**: `v1 - v2`는 `v1 + (-v2)`와 같습니다. 즉, `v2`의 방향을 뒤집은 `-v2` 벡터를 `v1`에 더하는 것과 같습니다.
"""

# %%
# 원점 (Origin)
origin = np.array([0, 0])
v2_neg = -v2

# 그래프에 표시될 모든 점들을 모아 축의 범위를 동적으로 설정합니다.
all_points_x_sub = [0, v1[0], v_sub[0]]
all_points_y_sub = [0, v1[1], v_sub[1]]

plt.figure()
plt.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='r', label=f'v1 = {v1}')
# -v2를 v1의 끝점에서 평행이동하여 표시합니다.
plt.quiver(*v1, *v2_neg, angles='xy', scale_units='xy', scale=1, color='b', label=f'-v2 (translated) = {v2_neg}')
plt.quiver(*origin, *v_sub, angles='xy', scale_units='xy', scale=1, color='g', label=f'v1 - v2 = {v_sub}')

# 동적으로 계산된 범위에 여백을 주어 x, y축 범위를 설정합니다.
plt.xlim(min(all_points_x_sub) - 1, max(all_points_x_sub) + 1)
plt.ylim(min(all_points_y_sub) - 1, max(all_points_y_sub) + 1)

plt.title('Vector Subtraction')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid()
plt.legend()
plt.gca().set_aspect('equal', adjustable='box') # x, y축의 비율을 동일하게 설정합니다.
plt.show()

# %% [markdown]
"""
#### 스칼라 곱

벡터에 숫자를 곱하면 모든 요소에 곱셈이 적용되는 것을 보여줍니다.
"""

# %%
# 벡터의 모든 요소에 스칼라 값을 곱합니다.
alpha = 3
v = np.array([1, 2, -3])
print(f"{alpha} * {v} = {alpha * v}")

# %% [markdown]
"""
### 선형 결합과 내적
"""

# %% [markdown]
"""
#### 선형 결합 (Linear Combination)
벡터의 스칼라 곱과 덧셈을 조합하여 선형 결합을 실습합니다. 이는 벡터 조작의 핵심 개념입니다.
"""

# %%
a1 = np.array([1, 2])
a2 = np.array([3, 0])

# 스칼라 계수를 정의합니다.
c1 = 2
c2 = -1.5

# a1과 a2 벡터를 스칼라배하여 더합니다.
b_combined = c1 * a1 + c2 * a2
print(f"{c1}*a1 + {c2}*a2 = {b_combined}")

# %% [markdown]
"""
#### 내적 (Inner Product / Dot Product)

`np.dot()` 함수나 `@` 연산자를 사용하여 내적을 계산합니다.
내적이 '가중합'을 계산하는 강력한 도구임을 예시로 설명합니다.
"""

# %%
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 두 벡터의 내적을 계산합니다.
dot_product = np.dot(a, b)  # 또는 a @ b
print(f"{a} 와 {b} 의 내적: {dot_product}")

# %% [markdown]
"""
#### 예시: 총 비용 계산하기
"""

# %%
prices = np.array([1500, 3000, 500])
quantities = np.array([2, 1, 3])

# 가격 벡터와 수량 벡터의 내적은 총비용과 같습니다.
total_cost = prices @ quantities
print(f"총 지불 비용: {total_cost}원")

# %% [markdown]
"""
### 벡터의 크기(놈)와 거리
"""

# %% [markdown]
"""
#### 놈 (Norm) 계산하기
`np.linalg.norm()`을 사용하여 벡터의 크기(유클리드 놈)를 계산합니다.
"""

# %%
# 피타고라스 삼조(3, 4, 5)를 이용한 예시
x = np.array([3, 4])
norm_x = np.linalg.norm(x)
print(f"벡터 x={x} 의 놈(크기): {norm_x}")

# %% [markdown]
"""
#### 벡터 간의 거리 (Distance)
두 벡터의 차에 놈을 계산하여 거리를 구합니다.
"""

# %%
a = np.array([1, 2])
b = np.array([4, 6])

# 두 벡터 사이의 유클리드 거리는 두 벡터의 차의 놈과 같습니다.
distance = np.linalg.norm(a - b)
print(f"벡터 a={a} 와 b={b} 사이의 거리: {distance}")