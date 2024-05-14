![image](https://github.com/jinuk0211/xlstm-implementation/assets/150532431/aae98134-b4e1-408d-a646-0e62548b1be2)
기존 lstm 문제
1. Inability to revise storage decisions.
exponential gating으로 해결
2. Limited storage capacities (정보가 scalar cell state)으로 압축되야함
Rare Token Prediction로 해결, 성능 문제는 matrix memory로
3. memory mixing으로 인한 병렬화 x

Extended LSTM
sLSTM 
scalar memory,  scalar update, and memory mixing, 

![image](https://github.com/jinuk0211/xlstm-implementation/assets/150532431/e67d7718-4fdd-4b76-af53-1c272ebe54e1)

mLSTM
matrix memory, covariance (outer product) update rule(완전히 병렬가능) 

![image](https://github.com/jinuk0211/xlstm-implementation/assets/150532431/823da186-ba03-46ac-9610-f6f5f788f522)

관련 연구
Linear Attention
State Space Models(mamba)
Recurrent Neural Networks(RNN)
Gating
Covariance Update Rule(RWKV)
