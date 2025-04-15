# 0. Nhập mã cổ phiếu từ người dùng
ticker = input("📥 Nhập mã cổ phiếu (ví dụ TSLA, AAPL, MSFT): ").upper()

# 1. Tải dữ liệu
import yfinance as yf

df = yf.download(ticker, period="90d", interval="1d")[["Close"]]

# 2. Chuẩn hóa dữ liệu
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 3. Tạo chuỗi thời gian cho LSTM
X = []
y = []
window_size = 10

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size : i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Mô hình LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=30, batch_size=8, verbose=0)

# 5. Dự đoán ngày mai
last_10_days = scaled_data[-window_size:]
last_10_days = np.reshape(last_10_days, (1, window_size, 1))
predicted_scaled = model.predict(last_10_days)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

# 6. In kết quả
today_price = df["Close"].iloc[-1]
print(f"\n✅ Mã cổ phiếu: {ticker}")
print(f"✅ Giá hôm nay: ${float(today_price):.2f}")
print(f"🔮 Dự đoán ngày mai: ${float(predicted_price):.2f}")

# 7. Biểu đồ
import matplotlib.pyplot as plt

plt.plot(df["Close"], label="Giá thực tế")
plt.axhline(y=predicted_price, color="red", linestyle="--", label="Dự đoán ngày mai")
plt.title(f"Dự đoán giá {ticker}")
plt.legend()
plt.grid(True)
plt.show()
