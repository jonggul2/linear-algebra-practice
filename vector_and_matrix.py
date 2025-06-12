# %% [markdown]
"""
# Vector and Matrix Manipulation Practice
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

# %% [markdown]
"""
---
## Part 1: 벡터 (Vectors)
---

벡터는 크기와 방향을 함께 가지는 양으로, 숫자를 순서대로 나열한 것입니다.
머신러닝에서는 데이터를 표현하는 가장 기본적인 단위로 사용됩니다.
"""

# %% [markdown]
"""
### 1.1. 벡터 생성과 기본 속성

#### 기본 벡터 생성하기
`np.array()` 함수에 리스트를 전달하여 간단하게 벡터(1차원 배열)를 만들 수 있습니다.
"""

# %%
# 실습: 아래 벡터의 값들을 다른 숫자로 바꾸고 실행해보세요.
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
print("벡터 a:", a)
print("벡터 b:", b)

# %% [markdown]
"""
#### 벡터의 속성 확인하기
- `.shape`: 벡터의 크기(몇 개의 요소가 있는지)를 알려줍니다.
- `.size`: 벡터에 포함된 총 요소의 개수를 알려줍니다. 1차원 벡터에서는 `.shape`의 값과 동일합니다.
"""

# %%
# 실습: 위에서 정의한 a, b 벡터의 속성을 확인합니다.
print("a의 형태(크기):", a.shape)
print("a의 요소 개수:", a.size)
print("---")
print("b의 형태(크기):", b.shape)
print("b의 요소 개수:", b.size)

# %% [markdown]
"""
#### 특정 요소에 접근하기 (Indexing)
벡터의 개별 요소에 접근할 때는 인덱싱을 사용합니다. **Python에서는 인덱스가 0부터 시작**한다는 점을 기억하세요!
"""

# %%
# 실습: a 벡터의 인덱스 1에 접근합니다.
print(f"a[1]: {a[1]}")

# 실습: b 벡터의 인덱스 2에 접근합니다.
print(f"b[2]: {b[2]}")

# %% [markdown]
"""
#### 특별한 형태의 벡터 만들기
때로는 모든 요소가 0 또는 1로 채워진 벡터가 필요합니다.
- `np.zeros()`: 영 벡터(모든 요소가 0)를 생성합니다.
- `np.ones()`: 일 벡터(모든 요소가 1)를 생성합니다.
"""

# %%
# 실습: 크기가 4인 영 벡터와 크기가 3인 일 벡터를 만들어보세요.
# 모든 요소가 0인 벡터를 생성합니다.
zero_vec = np.zeros(4)
print("영 벡터 (zeros):\n", zero_vec)
print("-" * 20)

# 모든 요소가 1인 벡터를 생성합니다.
ones_vec = np.ones(3)
print("1 벡터 (ones):\n", ones_vec)

# %% [markdown]
"""
### 1.2. 벡터의 기본 연산

#### 덧셈과 뺄셈
크기(차원)가 같은 두 벡터는 서로 더하거나 뺄 수 있습니다.
"""

# %%
# 실습: 아래 v1과 v2의 값을 다른 숫자로 바꾸고, 덧셈/뺄셈 결과와 그래프가 어떻게 변하는지 관찰해보세요.
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
벡터의 연산은 기하학적으로 표현할 수 있습니다. 벡터 `v1`과 `v2`의 덧셈은 **'꼬리-머리 이어붙이기'(Tip-to-Tail)** 방식으로 시각화됩니다.
- `v1` 벡터의 머리(끝점)에 `v2` 벡터의 꼬리(시작점)를 이어 붙이면, 그 결과는 원점에서 `v2`의 새로운 머리까지의 벡터 `v1 + v2`와 같습니다.
"""

# %%
# 원점 (Origin)
origin = np.array([0, 0])

# 그래프에 표시될 모든 점들을 모아 축의 범위를 동적으로 설정합니다.
all_points_x = [0, v1[0], v_add[0]]
all_points_y = [0, v1[1], v_add[1]]


plt.figure(figsize=(7,7))
# v1 벡터를 원점에서 시작하여 그립니다.
plt.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='r', label=f'v1 = {v1}')
# v2 벡터를 v1의 끝점에서 시작하여 그립니다 (평행이동).
plt.quiver(*v1, *v2, angles='xy', scale_units='xy', scale=1, color='b', label=f'v2 (translated) = {v2}')
# v1 + v2 결과 벡터를 원점에서 시작하여 그립니다.
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
벡터 `v1`에서 `v2`를 빼는 것은, `v1`에 `v2`의 반대 방향 벡터(`-v2`)를 더하는 것과 같습니다 (`v1 + (-v2)`).
덧셈과 마찬가지로 '꼬리-머리 이어붙이기' 방식으로 시각화할 수 있습니다.
"""

# %%
# 원점 (Origin)
origin = np.array([0, 0])
v2_neg = -v2

# 그래프에 표시될 모든 점들을 모아 축의 범위를 동적으로 설정합니다.
all_points_x_sub = [0, v1[0], v_sub[0]]
all_points_y_sub = [0, v1[1], v_sub[1]]

plt.figure(figsize=(7,7))
# v1 벡터를 원점에서 시작하여 그립니다.
plt.quiver(*origin, *v1, angles='xy', scale_units='xy', scale=1, color='r', label=f'v1 = {v1}')
# -v2 벡터를 v1의 끝점에서 시작하여 그립니다 (평행이동).
plt.quiver(*v1, *v2_neg, angles='xy', scale_units='xy', scale=1, color='b', label=f'-v2 (translated) = {v2_neg}')
# v1 - v2 결과 벡터를 원점에서 시작하여 그립니다.
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
#### 스칼라 곱 (Scalar Multiplication)
스칼라(단일 숫자)를 벡터에 곱하면, 벡터의 모든 요소에 해당 스칼라가 곱해집니다.
"""

# %%
# 실습: 스칼라 alpha의 값을 양수, 음수, 0으로 바꿔보며 결과 벡터가 어떻게 변하는지 확인해보세요.
alpha = 3
v = np.array([1, 2, -3])
print(f"{alpha} * {v} = {alpha * v}")

# %% [markdown]
"""
### 1.3. 벡터의 고급 연산: 선형 결합과 내적

#### 선형 결합 (Linear Combination)
여러 벡터에 스칼라를 곱한 뒤 더하여 새로운 벡터를 만드는 연산입니다.
"""

# %%
# 실습: c1, c2의 값을 바꿔보며 선형 결합의 결과가 어떻게 달라지는지 확인해보세요.
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
두 벡터의 각 요소별 곱의 총합입니다. `@` 또는 `np.dot()`으로 계산하며, '가중합' 계산 등에 활용됩니다.
"""

# %%
# 실습: a와 b 벡터의 값을 바꿔보며 내적 결과가 어떻게 변하는지 확인해보세요.
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 두 벡터의 내적을 계산합니다.
dot_product = np.dot(a, b)  # 또는 a @ b
print(f"{a} 와 {b} 의 내적: {dot_product}")

# %% [markdown]
"""
#### 내적 활용 예시: 총 구매 비용 계산하기
가격 벡터와 수량 벡터의 내적은 총 구매 비용과 같습니다.
"""

# %%
# 실습: 가격이나 수량 값을 바꿔서 총 비용을 다시 계산해보세요.
prices = np.array([1500, 3000, 500])
quantities = np.array([2, 1, 3])

# 가격 벡터와 수량 벡터의 내적은 총비용과 같습니다.
total_cost = prices @ quantities
print(f"총 지불 비용: {total_cost}원")

# %% [markdown]
"""
### 1.4. 벡터의 크기와 거리

#### 놈 (Norm) 계산하기: 벡터의 크기
벡터의 **놈(Norm)** 은 원점(0,0)에서 벡터의 끝점까지의 거리를 의미하며, 보통 벡터의 '크기'나 '길이'를 나타냅니다.
`np.linalg.norm()` 함수를 사용하여 유클리드 놈(L2 Norm)을 계산할 수 있습니다.
"""

# %%
# 실습: 벡터 x의 값을 바꿔보며 놈(크기) 값이 어떻게 변하는지 확인해보세요.
# 피타고라스 삼조(3, 4, 5)를 이용한 예시
x = np.array([3, 4])
norm_x = np.linalg.norm(x)
print(f"벡터 x={x} 의 놈(크기): {norm_x}")

# %% [markdown]
"""
#### 벡터 간의 거리 (Distance)
두 벡터(점) `a`와 `b` 사이의 거리는, 두 벡터의 차(`a - b`)의 놈(크기)과 같습니다.
즉, 한 점에서 다른 점으로 이동하는 벡터의 크기를 계산하는 것과 동일합니다.
"""

# %%
# 실습: 벡터 a와 b의 값을 바꿔보며 두 점 사이의 거리가 어떻게 계산되는지 확인해보세요.
a = np.array([1, 2])
b = np.array([4, 6])

# 두 벡터 사이의 유클리드 거리는 두 벡터의 차의 놈과 같습니다.
distance = np.linalg.norm(a - b)
print(f"벡터 a={a} 와 b={b} 사이의 거리: {distance}")

# %% [markdown]
"""
---
## Part 2: 행렬 (Matrices)
---

행렬은 숫자를 직사각형 격자 형태로 배열한 것으로, 벡터의 확장된 개념입니다.
데이터셋, 연립 방정식, 이미지 등 다양한 형태의 데이터를 표현하며, 데이터를 변환(transformation)하는 역할을 합니다.
"""

# %% [markdown]
"""
### 2.1. 행렬 생성 및 속성

#### Numpy로 행렬 생성하기
`np.array()` 함수에 2차원 리스트를 전달하여 행렬을 생성합니다.
"""

# %%
# 실습: 행렬의 값이나 크기를 바꾸고 실행해보세요.
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("행렬 A:\n", A)

# %% [markdown]
"""
#### 행렬의 속성 확인
- `.shape`: 행렬의 크기를 (행, 열) 형태로 알려줍니다.
- `.ndim`: 행렬의 차원을 알려줍니다. (행렬은 2차원)
- `A[행, 열]`: 특정 위치의 요소에 접근합니다. (인덱스는 0부터 시작)
"""

# %%
# 실습: 위에서 정의한 A 행렬의 속성을 확인하고, 특정 요소에 접근해보세요.
print(f"A.shape: {A.shape}")
print(f"A.ndim: {A.ndim}")
# A 행렬의 1행 2열 요소 (0-based-index)에 접근합니다.
print(f"A[0, 2]: {A[0, 2]}")


# %% [markdown]
"""
#### 특별한 형태의 행렬 만들기
- `np.zeros((행, 열))`: 모든 요소가 0인 영 행렬을 생성합니다.
- `np.identity(n)`: 주대각선이 1이고 나머지는 0인 `n x n` 크기의 단위 행렬(항등 행렬)을 생성합니다.
- `np.diag([...])`: 리스트를 주대각선 요소로 갖는 대각 행렬을 생성합니다.
- `np.random.rand(행, 열)`: 0과 1 사이의 무작위 값으로 채워진 행렬을 생성합니다.
"""

# %%
# 실습: 각 함수의 인자(shape, size 등)를 바꿔서 다양한 특별한 행렬을 만들어보세요.
# 모든 요소가 0인 2x3 행렬을 생성합니다.
Z = np.zeros((2, 3))
print("2x3 영 행렬 (zeros):\n", Z)
print("-" * 20)

# 3x3 단위 행렬을 생성합니다. (주대각선이 1이고 나머지는 0)
I = np.identity(3)
print("3x3 단위 행렬 (identity):\n", I)
print("-" * 20)

# 주대각선에 특정 값을 가지는 대각 행렬을 생성합니다.
D = np.diag([0.2, -3, 1.2])
print("대각 행렬 (diag):\n", D)
print("-" * 20)

# 0과 1 사이의 무작위 값으로 채워진 2x2 행렬을 생성합니다.
R = np.random.rand(2, 2)
print("2x2 랜덤 행렬 (random):\n", R)

# %% [markdown]
"""
#### 행렬의 전치 (Transpose)
전치 행렬은 원본 행렬의 행과 열을 서로 맞바꾼 행렬입니다. `.T` 속성으로 간단히 구할 수 있습니다.
(m, n) 크기 행렬의 전치 행렬은 (n, m) 크기가 됩니다.
"""

# %%
# 실습: 아래 A 행렬을 전치시키고, 원본과 모양(shape)을 비교해보세요.
A_T = A.T
print("원본 행렬 A:\n", A)
print(f"A.shape: {A.shape}")
print("-" * 20)
print("A의 전치 행렬 A.T:\n", A_T)
print(f"A.T.shape: {A_T.shape}")

# %% [markdown]
"""
### 2.2. 행렬의 기본 연산

#### 덧셈, 뺄셈, 스칼라 곱
벡터와 마찬가지로, 크기가 같은 행렬끼리는 요소별 덧셈과 뺄셈이 가능하며, 스칼라 곱 또한 모든 요소에 적용됩니다.
"""

# %%
# 실습: 행렬의 값을 바꾸거나, 스칼라 값을 바꿔서 연산 결과를 확인해보세요.
# 연산을 쉽게 확인하기 위해 간단한 정수 행렬을 새로 정의합니다.
A_add = np.array([
    [10, 20, 30],
    [40, 50, 60]
])
B_add = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("행렬 A:\n", A_add)
print("행렬 B:\n", B_add)
print("-" * 20)

# 덧셈
print("A + B:\n", A_add + B_add)
print("-" * 20)

# 뺄셈
print("A - B:\n", A_add - B_add)
print("-" * 20)

# 스칼라 곱
print("2 * A:\n", 2 * A_add)

# %% [markdown]
"""
### 2.3. 행렬 곱셈과 그 활용

#### 행렬 곱셈 (Matrix-Matrix Multiplication)
두 행렬의 곱셈 `C = AB`는 첫 번째 행렬 `A`의 열 개수와 두 번째 행렬 `B`의 행 개수가 같아야 가능합니다.
(m, p) 크기 행렬과 (p, n) 크기 행렬을 곱하면 결과는 (m, n) 크기의 행렬이 됩니다.
행렬 곱은 **선형 변환의 연속(합성)**을 의미하며, 교환 법칙(`AB != BA`)이 성립하지 않는다는 특징이 있습니다.
"""

# %%
# 실습: 두 행렬의 값을 바꿔보거나, 행렬의 크기를 바꿔서 곱셈을 시도해보세요.
# (m, p) x (p, n) 크기의 행렬 곱셈
A_mul = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
B_mul = np.array([
    [10, 11],
    [20, 21],
    [30, 31]
])

# 행렬 곱셈. A_mul의 열 수(3)와 B_mul의 행 수(3)가 같아야 합니다.
C_mul = A_mul @ B_mul  # 결과는 (2, 2) 행렬

print("행렬 A (2x3):\n", A_mul)
print("행렬 B (3x2):\n", B_mul)
print("-" * 20)
print("행렬 곱 AB:\n", C_mul)
print(f"AB의 크기: {C_mul.shape}")

# %% [markdown]
"""
#### 벡터의 외적 (Outer Product)
두 벡터 `a`(크기 m)와 `b`(크기 n)의 외적은 `m x n` 크기의 행렬을 생성합니다.
`a`의 각 요소와 `b`의 각 요소의 모든 조합을 곱한 행렬로, `np.outer(a, b)`로 계산합니다.
외적은 두 벡터의 상호작용을 행렬로 표현하여, 데이터 분석이나 머신러닝 모델의 가중치 표현 등에서 활용됩니다.
"""

# %%
# 실습: a_vec, b_vec의 값이나 크기를 바꿔보며 외적의 결과가 어떻게 변하는지 확인해보세요.
a_vec = np.array([1, 2, 3])
b_vec = np.array([10, 20])

# np.outer() 함수를 사용하여 두 벡터의 외적을 계산합니다.
outer_product = np.outer(a_vec, b_vec)

print("벡터 a:", a_vec)
print("벡터 b:", b_vec)
print("-" * 20)
print("a와 b의 외적 (np.outer(a,b)):\n", outer_product)
print(f"외적 행렬의 크기: {outer_product.shape}")

# %% [markdown]
"""
#### 활용 1: 기하 변환 (Geometric Transformations)
행렬 곱셈은 벡터를 특정 방식으로 변환하는 강력한 도구입니다. 예를 들어, 2D 벡터를 `θ`만큼 **회전**시키는 변환은 아래와 같은 회전 행렬 `R`을 곱하여 수행할 수 있습니다.

```
      [ cos(θ)  -sin(θ) ]
R =   [ sin(θ)   cos(θ) ]
```

`변환된 벡터 = R @ 원본 벡터`
"""

# %%
# 실습: 벡터 v의 값이나, 회전 각도 theta를 바꿔서 변환 결과를 확인해보세요.
# 변환할 2D 벡터를 정의합니다.
v = np.array([1, 0]) # x축 방향의 단위 벡터

# 45도 회전을 위한 변환 행렬을 생성합니다.
theta = np.radians(45)
R_mat = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# 회전 변환을 적용합니다.
v_rotated = R_mat @ v

print("원본 벡터 v:", v)
print("45도 회전 행렬 R:\n", np.round(R_mat, 2))
print("회전된 벡터 v_rotated:", np.round(v_rotated, 2))

# %%
# 시각화
origin = np.array([0, 0])

all_points_x_rot = [0, v[0], v_rotated[0]]
all_points_y_rot = [0, v[1], v_rotated[1]]

plt.figure(figsize=(6,6))
plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color='r', label=f'Original v = {v}')
plt.quiver(*origin, *v_rotated, angles='xy', scale_units='xy', scale=1, color='g', label=f'Rotated v = ({v_rotated[0]:.2f}, {v_rotated[1]:.2f})')

# 동적으로 계산된 범위에 여백을 주어 x, y축 범위를 설정합니다.
plt.xlim(min(all_points_x_rot) - 1, max(all_points_x_rot) + 1)
plt.ylim(min(all_points_y_rot) - 1, max(all_points_y_rot) + 1)

plt.title('Geometric Transformation (Rotation)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid()
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# %% [markdown]
"""
#### 활용 2: 선형 변환의 합성 (Composition)
두 가지 선형 변환을 연속으로 적용하는 것은, 각 변환에 해당하는 두 행렬을 먼저 곱하여 얻은 **합성 행렬**을 한 번 적용하는 것과 같습니다.
예를 들어, 30도 회전 후 60도 회전을 적용하는 것은, 90도 회전 행렬을 한 번 적용하는 것과 동일합니다.

`(R_60 @ R_30) @ v = R_60 @ (R_30 @ v)`
"""

# %%
# 실습: 벡터 v_comp의 값이나, 두 회전 각도를 바꿔보며 결과를 확인해보세요.
# 원본 벡터를 정의합니다.
v_comp = np.array([2, 1])

# 30도 회전 변환
theta_30 = np.radians(30)
R_30 = np.array([[np.cos(theta_30), -np.sin(theta_30)],
                 [np.sin(theta_30),  np.cos(theta_30)]])

# 60도 회전 변환
theta_60 = np.radians(60)
R_60 = np.array([[np.cos(theta_60), -np.sin(theta_60)],
                 [np.sin(theta_60),  np.cos(theta_60)]])

# 방법 1: 변환을 순차적으로 적용합니다 (30도 회전 후 60도 회전).
v_rotated_seq = R_60 @ (R_30 @ v_comp)

# 방법 2: 변환 행렬을 먼저 곱하여 합성한 후, 한 번에 적용합니다.
R_90 = R_60 @ R_30
v_rotated_combined = R_90 @ v_comp

print("원본 벡터:", v_comp)
print("-" * 20)
print(f"순차 변환 결과 (30도 -> 60도):\n", np.round(v_rotated_seq, 2))
print(f"합성 변환 결과 (90도):\n", np.round(v_rotated_combined, 2))

# %% [markdown]
"""
### 2.4. 역행렬과 행렬식

#### 역행렬 (Matrix Inverse)
어떤 정방 행렬(square matrix) `A`에 대해, 곱했을 때 단위 행렬(`I`)이 되는 행렬 `B`가 존재한다면, `B`를 `A`의 **역행렬(inverse matrix)**이라 부르고 `A⁻¹`로 표기합니다.

`A @ A⁻¹ = A⁻¹ @ A = I`

역행렬은 어떤 변환을 '되돌리는'(undo) 변환에 해당하며, 연립 방정식을 푸는 데 핵심적인 역할을 합니다.
**모든 정방 행렬이 역행렬을 갖는 것은 아닙니다.**

- `np.linalg.inv()` 함수로 역행렬을 계산할 수 있습니다.
"""

# %%
# 실습: 아래 행렬 A_inv_source의 값을 바꿔보며 역행렬을 계산해보세요.
# 역행렬을 계산할 2x2 정방 행렬 정의
A_inv_source = np.array([
    [1, 1],
    [1, 2]
])

# 역행렬 계산
try:
    A_inverse = np.linalg.inv(A_inv_source)
    print("원본 행렬 A:\n", A_inv_source)
    print("-" * 20)
    print("A의 역행렬 A⁻¹:\n", A_inverse)
    print("-" * 20)

    # A @ A⁻¹가 단위 행렬인지 확인 (부동소수점 오차를 고려)
    identity_check = A_inv_source @ A_inverse
    print("A @ A⁻¹ 결과:\n", identity_check)

    # np.allclose()를 이용한 단위 행렬 검증
    is_identity = np.allclose(identity_check, np.identity(2))
    print(f"\n결과가 단위 행렬과 매우 가깝습니까? -> {is_identity}")

except np.linalg.LinAlgError as e:
    print("오류:", e)
    print("이 행렬은 역행렬을 가지지 않습니다 (특이 행렬).")

# %% [markdown]
"""
#### 행렬식 (Determinant)
행렬식은 정방 행렬이 갖는 고유한 스칼라 값으로, `np.linalg.det()`으로 계산합니다.
기하학적으로 행렬이 변환시키는 공간의 '부피'가 얼마나 변하는지를 나타냅니다.

- **`det(A) ≠ 0`**: 행렬 `A`는 역행렬을 가집니다 (가역 행렬, Invertible).
- **`det(A) = 0`**: 행렬 `A`는 역행렬을 가지지 않습니다 (특이 행렬, Singular).

따라서 행렬식은 역행렬의 존재 여부를 판별하는 중요한 지표입니다.
"""

# %%
# 실습: A_inv_source의 행렬식을 계산해보고, 아래 특이 행렬의 행렬식과 비교해보세요.
# 가역 행렬 (Invertible Matrix)
det_A = np.linalg.det(A_inv_source)
print(f"가역 행렬 A:\n{A_inv_source}")
print(f"A의 행렬식: {det_A:.1f}")
print("-" * 20)

# 특이 행렬 (Singular Matrix)
singular_matrix_det_check = np.array([
    [1, 2],
    [2, 4]
])
det_S = np.linalg.det(singular_matrix_det_check)
print(f"특이 행렬 S:\n{singular_matrix_det_check}")
print(f"S의 행렬식: {det_S:.1f}")

# %% [markdown]
"""
#### 역행렬 계산 시도와 특이 행렬
역행렬이 존재하지 않는 **특이 행렬(Singular Matrix)**에 `np.linalg.inv()`를 사용하면 `LinAlgError`가 발생합니다.
"""

# %%
# 실습: 특이 행렬에 역행렬 계산을 시도하면 LinAlgError가 발생하는 것을 확인합니다.
singular_matrix = np.array([
    [1, 2],
    [2, 4]
])

print("특이 행렬 B:\n", singular_matrix)
print("-" * 20)

# 역행렬 계산 시도
try:
    B_inverse = np.linalg.inv(singular_matrix)
    print("B의 역행렬:\n", B_inverse)
except np.linalg.LinAlgError as e:
    print("np.linalg.inv(B) 실행 시 오류 발생:")
    print(f"-> {e}")