# %% [markdown]
"""
# Matrix Manipulation Practice

**목표**: 행렬의 개념을 코드로 이해하고, 기본 연산과 행렬-벡터 곱의 의미를 실습을 통해 익힙니다.
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
### 행렬 생성 및 속성 확인
"""

# %% [markdown]
"""
#### Numpy로 행렬 생성하기
`np.array()`에 2차원 리스트를 전달하여 행렬을 생성합니다.
"""

# %%
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print("행렬 A:\n", A)

# %% [markdown]
"""
#### 행렬 속성 확인
`.shape` (크기), `.ndim` (차원), 특정 요소 접근(indexing)을 실습합니다.
"""

# %%
print(f"A.shape: {A.shape}")
print(f"A.ndim: {A.ndim}")
# A 행렬의 (0, 2) 인덱스에 접근합니다.
print(f"A[0, 2]: {A[0, 2]}")


# %% [markdown]
"""
#### 특별한 행렬 만들기
영 행렬(`np.zeros`), 단위 행렬(`np.identity`), 대각 행렬(`np.diag`), 랜덤 행렬(`np.random.rand`)을 만듭니다.
"""

# %%
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
`.T` 속성을 사용해 행과 열을 바꾼 전치행렬을 구합니다.
"""

# %%
A_T = A.T
print("원본 행렬 A:\n", A)
print("-" * 20)
print("A의 전치 행렬 A.T:\n", A_T)
print("-" * 20)
print(f"A.T.shape: {A_T.shape}")

# %% [markdown]
"""
### 행렬의 기본 연산
"""

# %% [markdown]
"""
#### 덧셈, 뺄셈, 스칼라 곱
같은 크기의 행렬끼리 덧셈, 뺄셈을 수행하고, 행렬에 스칼라 값을 곱합니다.
"""

# %%
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
#### 행렬-벡터 곱 (Matrix-Vector Multiplication)
행렬과 벡터의 곱셈을 실습하고, 두 가지 관점에서 의미를 해석합니다.
"""

# %%
x = np.array([10, 20, 30])
y = A @ x  # np.dot(A, x) 와 동일
print("벡터 x:", x)
print("-" * 20)
print("행렬-벡터 곱 Ax 결과 벡터 y:\n", y)

# %% [markdown]
"""
##### 행(Row) 관점
결과 벡터의 각 요소는 행렬의 각 행과 입력 벡터의 내적으로 계산됩니다.
"""

# %%
# y[0]은 A의 첫 번째 행과 x의 내적과 같습니다.
y0_check = A[0, :] @ x
print(f"y[0] 계산: {A[0, :]} @ {x} = {y0_check}")
# y[1]은 A의 두 번째 행과 x의 내적과 같습니다.
y1_check = A[1, :] @ x
print(f"y[1] 계산: {A[1, :]} @ {x} = {y1_check}")

# %% [markdown]
"""
##### 열(Column) 관점
결과 벡터는 행렬의 열벡터들을 입력 벡터의 요소로 선형 결합한 것과 같습니다.
"""

# %%
# y는 A의 열벡터들의 선형 결합입니다: x[0]*A[:,0] + x[1]*A[:,1] + ...
y_check_col = x[0] * A[:, 0] + x[1] * A[:, 1] + x[2] * A[:, 2]
print("y (열 관점 체크):", y_check_col)

# %% [markdown]
"""
### 행렬 곱셈과 활용
"""

# %% [markdown]
"""
#### 행렬 곱셈 (Matrix-Matrix Multiplication)
`@` 연산자나 `np.dot()`을 이용해 행렬 간의 곱셈을 수행합니다.
(m, p) x (p, n) 크기의 행렬을 곱하면 (m, n) 크기의 행렬이 됨을 확인합니다.
"""

# %%
# (2, 3) 행렬과 (3, 2) 행렬의 곱셈
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
두 벡터의 외적(`a @ b^T`)은 행렬을 생성합니다. 이는 `m`-벡터 `a`와 `n`-벡터 `b`를 곱하여 `m x n` 행렬을 만드는 중요한 연산입니다.

  ```
        [ a1*b1   a1*b2  ...  a1*bn  ]
  ab^T =  [ a2*b1   a2*b2  ...  a2*bn  ]
        [  ...     ...    ...   ...   ]
        [ am*b1   am*b2  ...  am*bn  ]
  ```
"""

# %%
# 외적 (Outer Product) 예제
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
#### 기하 변환 (Geometric Transformations)
많은 2D 및 3D 벡터의 기하학적 변환과 매핑은 행렬 곱셈 `y = Ax`로 표현할 수 있습니다.

예를 들어, 벡터를 **`θ`만큼 회전**시키는 변환은 다음과 같은 회전 행렬 `A`를 사용하여 나타낼 수 있습니다.

```
      [ cos(θ)  -sin(θ) ]
A =   [ sin(θ)   cos(θ) ]
```

결과 벡터 `y`는 `y = Ax`로 계산됩니다.
"""

# %%
# 변환할 2D 벡터를 정의합니다.
v = np.array([1, 0]) # x축 방향의 단위 벡터

# 45도 회전을 위한 변환 행렬을 생성합니다.
theta = np.radians(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# 회전 변환을 적용합니다.
v_rotated = R @ v

print("원본 벡터 v:", v)
print("45도 회전 행렬 R:\n", np.round(R, 2))
print("회전된 벡터 v_rotated:", np.round(v_rotated, 2))

# %%
# 시각화
origin = np.array([0, 0])

all_points_x = [0, v[0], v_rotated[0]]
all_points_y = [0, v[1], v_rotated[1]]

plt.figure()
plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color='r', label=f'Original v = {v}')
plt.quiver(*origin, *v_rotated, angles='xy', scale_units='xy', scale=1, color='g', label=f'Rotated v = ({v_rotated[0]:.2f}, {v_rotated[1]:.2f})')

# 동적으로 계산된 범위에 여백을 주어 x, y축 범위를 설정합니다.
plt.xlim(min(all_points_x) - 1, max(all_points_x) + 1)
plt.ylim(min(all_points_y) - 1, max(all_points_y) + 1)

plt.title('Geometric Transformation (Rotation)')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.grid()
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# %% [markdown]
"""
#### 선형 변환의 합성 (Composition)
두 선형 변환을 연속으로 적용하는 것은, 두 변환 행렬을 곱한 새로운 행렬로 한 번에 변환하는 것과 같습니다.

이전의 기하 변환 예시를 활용하여, 30도 회전 후 60도 회전을 추가로 적용하는 것이 90도 회전과 같은지 확인해봅니다.
"""

# %%
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
### 역행렬 (Matrix Inverse)
정방 행렬(square matrix) `A`에 대해, 다음을 만족하는 행렬 `B`가 존재한다면 `B`를 `A`의 역행렬(inverse matrix)이라고 부르고 `A⁻¹`로 표기합니다.

`A @ A⁻¹ = A⁻¹ @ A = I`

여기서 `I`는 단위 행렬(identity matrix)입니다.

- 오직 정방 행렬 중에서도 특이 행렬(singular matrix)이 아닌, 즉 **가역 행렬(invertible matrix)**만이 역행렬을 가집니다.
- Numpy에서는 `np.linalg.inv()` 함수를 사용하여 역행렬을 계산할 수 있습니다.
"""

# %%
# 역행렬을 계산할 2x2 정방 행렬 정의 (피보나치 수열 응용)
A_inv_source = np.array([
    [1, 1],
    [1, 2]
])

# 역행렬 계산 (가역 행렬)
try:
    A_inverse = np.linalg.inv(A_inv_source)
    print("원본 행렬 A:\n", A_inv_source)
    print("-" * 20)
    print("A의 역행렬 A⁻¹:\n", A_inverse)
    print("-" * 20)
    
    # A @ A⁻¹가 단위 행렬인지 확인 (부동소수점 오차 감안)
    identity_check = A_inv_source @ A_inverse
    print("A @ A⁻¹ (단위 행렬 확인):\n", np.round(identity_check, decimals=10))

except np.linalg.LinAlgError as e:
    print("오류:", e)
    print("이 행렬은 역행렬을 가지지 않습니다 (특이 행렬).")

# %% [markdown]
"""
##### 특이 행렬 (Singular Matrix)의 경우
역행렬이 존재하지 않는 행렬(예: 한 행/열이 다른 행/열의 선형 조합이거나, 행렬식이 0인 경우)에 `np.linalg.inv()`를 사용하면 `LinAlgError`가 발생합니다.
"""

# %%
# 특이 행렬 (역행렬이 존재하지 않는 행렬)
# 2행이 1행의 2배이므로 선형 종속입니다.
singular_matrix = np.array([
    [1, 2],
    [2, 4]
])

print("특이 행렬 B:\n", singular_matrix)
print("-" * 20)

try:
    B_inverse = np.linalg.inv(singular_matrix)
    print("B의 역행렬:\n", B_inverse)
except np.linalg.LinAlgError as e:
    print("np.linalg.inv(B) 실행 시 오류 발생:")
    print(e) 