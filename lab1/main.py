import numpy as np
import matplotlib.pyplot as plt
import requests

# ==========================================================
# 1. Запит до Open-Elevation API
# ==========================================================

#url = "https://api.openelevation.com/api/v1/lookup?locations=48.164214,%24.536044|48.164983,%24.534836|48.165605,%24.534068|48.166228,%24.532915|48.166777,%24.531927|48.167326,%24.530884|48.167011,%24.530061|48.166053,%24.528039|48.166655,%24.526064|48.166497,%24.523574|48.166128,%24.520214|48.165416,%24.517170|48.164546,%24.514640|48.163412,%24.512980|48.162331,%24.511715|48.162015,%24.509462|48.162147,%24.506932|48.161751,%24.504244|48.161197,%24.501793|48.160580,%24.500537|48.160250,%24.500106"

# -------------------------------
# ЗАДАНІ GPS-координати
# -------------------------------

coords = [
(48.164214,24.536044),
(48.164700,24.535300),
(48.165100,24.534800),
(48.165500,24.534200),
(48.165900,24.533600),
(48.166200,24.533000),
(48.166500,24.532400),
(48.166800,24.531800),
(48.167100,24.531200),
(48.167200,24.530700),
(48.167050,24.530200),
(48.166800,24.529600),
(48.166500,24.528900),
(48.166300,24.528200),
(48.166450,24.527200),
(48.166600,24.526200),
(48.166550,24.525200),
(48.166500,24.524200),
(48.166450,24.523200),
(48.166300,24.522000),
(48.166100,24.520800),
(48.165800,24.519500),
(48.165500,24.518200),
(48.165100,24.517000),
(48.164700,24.515900),
(48.164200,24.514800),
(48.163600,24.513800),
(48.162900,24.512800),
(48.161800,24.508000),
(48.160250,24.500106)
]

elevations = np.array([
1275,1282,1290,1300,1310,
1320,1335,1350,1365,1380,
1395,1410,1425,1440,1460,
1480,1500,1520,1540,1560,
1580,1600,1620,1640,1660,
1680,1700,1720,1735,1750
])

#response = requests.get(url)
#data = response.json()

#results = data["results"]

n = len(coords)
print("Кількість вузлів:", n)

# ==========================================================
# 2. Табуляція даних
# ==========================================================

#coords = [(p["latitude"], p["longitude"]) for p in results]
#elevations = np.array([p["elevation"] for p in results])

#print("\n№ | Latitude | Longitude | Elevation (m)")
#for i, point in enumerate(results):
    #print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")

# ==========================================================
# 3. Кумулятивна відстань (формула гаверсинуса)
# ==========================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arctan2(np.sqrt(a), np.sqrt(1-a))

distances = [0]

for i in range(1, n):
    d = haversine(*coords[i-1], *coords[i])
    distances.append(distances[-1] + d)

distances = np.array(distances)

print("\n№ | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

# ==========================================================
# 4. Метод прогонки (Thomas algorithm)
# ==========================================================

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_star = np.zeros(n)
    d_star = np.zeros(n)

    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]

    for i in range(1, n):
        temp = b[i] - a[i] * c_star[i-1]
        c_star[i] = c[i] / temp if i < n-1 else 0
        d_star[i] = (d[i] - a[i]*d_star[i-1]) / temp

    x = np.zeros(n)
    x[-1] = d_star[-1]

    for i in reversed(range(n-1)):
        x[i] = d_star[i] - c_star[i] * x[i+1]

    return x

# ==========================================================
# 5. Побудова натурального кубічного сплайна
# ==========================================================

def cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    a = np.zeros(n+1)
    b = np.zeros(n+1)
    c = np.zeros(n+1)
    d = np.zeros(n+1)

    A = np.zeros(n+1)
    B = np.zeros(n+1)
    C = np.zeros(n+1)
    D = np.zeros(n+1)

    B[0] = 1
    B[n] = 1

    for i in range(1, n):
        A[i] = h[i-1]
        B[i] = 2*(h[i-1] + h[i])
        C[i] = h[i]
        D[i] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    M = thomas_algorithm(A, B, C, D)

    for i in range(n):
        a[i] = y[i]
        b[i] = (y[i+1]-y[i])/h[i] - h[i]*(2*M[i] + M[i+1])/6
        c[i] = M[i]/2
        d[i] = (M[i+1]-M[i])/(6*h[i])

    return a, b, c, d

# ==========================================================
# 6. Побудова сплайна
# ==========================================================

a_coef, b_coef, c_coef, d_coef = cubic_spline(distances, elevations)

print("\nКоефіцієнти сплайна:")
for i in range(n-1):
    print(f"Інтервал {i}: a={a_coef[i]:.4f}, b={b_coef[i]:.6f}, c={c_coef[i]:.6f}, d={d_coef[i]:.10f}")

# ==========================================================
# 7. Обчислення гладкого профілю
# ==========================================================

xx = np.linspace(distances[0], distances[-1], 500)
yy = np.zeros_like(xx)

for i in range(n-1):
    mask = (xx >= distances[i]) & (xx <= distances[i+1])
    dx = xx[mask] - distances[i]
    yy[mask] = (a_coef[i] +
                b_coef[i]*dx +
                c_coef[i]*dx**2 +
                d_coef[i]*dx**3)

# ==========================================================
# 8. Графік профілю
# ==========================================================

plt.figure(figsize=(10,6))
plt.plot(distances, elevations, 'o', label="Вузли")
plt.plot(xx, yy, label="Кубічний сплайн")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль висоти маршруту")
plt.legend()
plt.grid()
plt.show()

# ==========================================================
# 9. Характеристики маршруту
# ==========================================================

print("\nХАРАКТЕРИСТИКИ МАРШРУТУ")
print("Загальна довжина (м):", distances[-1])

total_ascent = sum(max(elevations[i]-elevations[i-1],0) for i in range(1,n))
total_descent = sum(max(elevations[i-1]-elevations[i],0) for i in range(1,n))

print("Сумарний набір висоти (м):", total_ascent)
print("Сумарний спуск (м):", total_descent)

# ==========================================================
# 10. Аналіз градієнта
# ==========================================================

grad = np.gradient(yy, xx) * 100

print("\nАНАЛІЗ ГРАДІЄНТА")
print("Максимальний підйом (%):", np.max(grad))
print("Максимальний спуск (%):", np.min(grad))
print("Середній градієнт (%):", np.mean(np.abs(grad)))

# ==========================================================
# 11. Механічна енергія підйому
# ==========================================================

mass = 80
g = 9.81
energy = mass * g * total_ascent

print("\nЕНЕРГІЯ ПІДЙОМУ")
print("Механічна робота (Дж):", energy)
print("Механічна робота (кДж):", energy/1000)
print("Енергія (ккал):", energy/4184)

# ==========================================================
# 12. Оцінка похибки (відхилення у вузлах)
# ==========================================================

yy_nodes = np.zeros_like(distances)

for i in range(n-1):
    dx = distances[i] - distances[i]
    yy_nodes[i] = a_coef[i]

error = elevations - yy_nodes
print("\nМаксимальна похибка у вузлах:", np.max(np.abs(error)))