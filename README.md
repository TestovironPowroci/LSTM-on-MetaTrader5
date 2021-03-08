# LSTM-on-MetaTrader5
Long Short Term Memory neural network trading on MetaTrader5 (Demo/Idea)


1. Create training data by training.mq5 using backtesting option. (ready data.csv in the folder is just an example)

2. Create testing data by training.mq5 using backtesting option. (test data should not be the same as training data)

3. Create validation data by training in pips.mq5 using backtesting option.

4. Finally train the neural network by LSTMtrainning.py.

5. When effects of training will be good enough(acording to the results of the neural network on test and validation data), save classifier.pth and optimizer.pth.

6. In MetaTrader5 load controling position.mq5 and live input data.mq5 on chart.

7. Turn on "One-Click Trading".
 
![one click trading](https://user-images.githubusercontent.com/79338815/110393510-f0d03e00-806a-11eb-8986-9d77304ffb68.JPG)

8. Write correct X AND Y coords of buy and sell buttons. (Use a mousotron or something similar for this.)

![cordina](https://user-images.githubusercontent.com/79338815/110393646-35f47000-806b-11eb-90ef-8c9e9af05b2e.JPG)

9. Find path of live data.csv. (this is a file with inputs, created by live input data.mq5 updated every 60 seconds) 

10. Find memory address of value 12332100 in MetaTrader5. (Use to that cheat engine)

11. Put path and memory address here.

![variables to change](https://user-images.githubusercontent.com/79338815/110396929-22e49e80-8071-11eb-8dcf-55a5643d125e.JPG)

12. Run LSTMlive.py and put MetaTrader5 on desktop.



Now LSTM nn is working.
