Enter stock symbol then the code will try to predict it fetches the data from yahoo for history then use the technical analyses such as MA MACD RSI from the past....to predict the stock 

<CR>
TSLA <BR>
Today 252.25 <BR>
Tomorrow 247.00 <BR>
Need to execute before pre market open 7 AM This is for my own use for testing. Please using it as reference only
![image](https://github.com/user-attachments/assets/ba3e22fd-cde4-4def-8efb-61d4d924d911)

Result is very close
![image](https://github.com/user-attachments/assets/3c666bcf-1a20-42bb-8b8c-bcc662c76a7d)

To run open 
https://colab.research.google.com
and paste the above code and hit play button <!
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

Result is here
![image](https://github.com/user-attachments/assets/f56dd433-067e-4ec8-870e-b5b3bb7b2107)



 Google Codelab vs Google Colab
Feature	Google Codelab	Google Colab
🔍 What is it?	Interactive tutorial website	Cloud-based Jupyter notebook (for running Python)
📚 Purpose	Teach you how to build apps, ML models, etc.	Run and experiment with Python code
🛠️ Can run code?	❌ No — it's just a step-by-step guide	✅ Yes — executes Python code in the cloud
🔗 Common Use	Links to GitHub, Colab, or downloads	Linked from Codelabs for live coding
📝 Content Format	Markdown-based tutorial	Python + Markdown cells (like Jupyter)
🌐 Where to use?	codelabs.developers.google.com	colab.research.google.com
👤 Account Needed?	No (but optional for saving progress)	Yes (Google account needed)

