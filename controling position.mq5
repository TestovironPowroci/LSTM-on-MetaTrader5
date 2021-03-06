//+------------------------------------------------------------------+
//|                                          controling position.mq5 |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>
CTrade          trade;
#include <Trade\PositionInfo.mqh>
CPositionInfo position;
  

string symbol1="DE30";

double x;
int c=0;
double genezis=0;
int odliczanie=0;

double y1;
double z1;
double p;

bool openPosition;
long positionType; 
 
int u =1; 
 
double ask;
double bid;
double slsp;
double slbp;
double sls;
double slb;
double tps;
double tpb;

 datetime ThisBar;
static datetime LastBar = 0;
int bylo=0;


double profit_pips()
{
      long symbol_digits=SymbolInfoInteger(position.Symbol(),SYMBOL_DIGITS);
      double difference;
      position.Select(symbol1);
      if(position.PositionType()==POSITION_TYPE_BUY)
      difference=position.PriceCurrent()-position.PriceOpen();
      else
      difference=position.PriceOpen()-position.PriceCurrent();
      double profit_p=(double)((difference*MathPow(10,symbol_digits))/100);
 return (profit_p);   
}


void  OnInit()

{
//ResetLastError();

//filehandle=FileOpen("dax30.csv",FILE_WRITE|FILE_CSV);
  // if(filehandle!=INVALID_HANDLE)
    // {      
     // Print("File opened correctly");
    // }
   //else Print("Error in opening file,",GetLastError());

   //return;

 

   
}





//digit = _Digits == 3 || _Digits == 5 ? _Point * 10 : _Point;
//Print("digit=",digit);

 void OnTick()
{
u=0;
openPosition=PositionSelect(symbol1);
positionType=(ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE); 
 
 
    MqlRates bar5[];
ArraySetAsSeries(bar5,true);
CopyRates(symbol1,PERIOD_M5,0,21,bar5);
   

 MqlRates bar1[];
 ArraySetAsSeries(bar1,true);
 CopyRates(symbol1,PERIOD_M1,0,61,bar1);

MqlRates barD1[];
ArraySetAsSeries(barD1,true);
CopyRates(symbol1,PERIOD_D1,0,2,barD1);



ask=SymbolInfoDouble(symbol1,SYMBOL_ASK);
bid=SymbolInfoDouble(symbol1,SYMBOL_BID);
slsp=bid+300;
slbp=ask-300;
sls =bid+2;
slb =ask-2;
tps =bid-80;
tpb =ask+80;

  


if(openPosition==false)
{
if (genezis==0)
{
p=987654321;
}
else
{
p=12332101;
}


if (genezis==0)
{
Sleep(60000);
genezis=1;
}

}

if(openPosition==true)
{
p=32112300;





if(positionType == POSITION_TYPE_SELL)
 {
 Print("sell");
 trade.PositionModify(symbol1,slsp,tps);
 
 ThisBar = (datetime)SeriesInfoInteger(symbol1,PERIOD_M5,SERIES_LASTBAR_DATE);
if(LastBar != ThisBar)
  {
   
   LastBar = ThisBar;
   odliczanie=odliczanie+1;
   
   MqlRates bar1ss[];
ArraySetAsSeries(bar1ss,true);
CopyRates(symbol1,PERIOD_M5,0,7,bar1ss); 
   
   



  
   position.Select(symbol1);
   Print(bar1ss[1].open,"|<|",bar1ss[1].close,"|<|",position.PriceOpen(),"||||",profit_pips());
   if((bar1ss[1].open<=bar1ss[1].close) && (bar1ss[1].close<position.PriceOpen()) && (profit_pips()>0))
   {
   Print("sell end");
   trade.PositionClose(symbol1);
   odliczanie=0;
   
   }
   if(odliczanie>=7)
   {
   if(bar1ss[1].close>=bar1ss[1].open || bar1ss[2].close>=bar1ss[2].open || bar1ss[1].close>=bar1ss[1].open)
   {
   trade.PositionClose(symbol1);
   odliczanie=0;
   }
   }
   }
 }
 
 if(positionType == POSITION_TYPE_BUY)
 {
 Print("buy");
 trade.PositionModify(symbol1,slbp,tpb);

  ThisBar = (datetime)SeriesInfoInteger(symbol1,PERIOD_M5,SERIES_LASTBAR_DATE);
if(LastBar != ThisBar)
  {
   
   LastBar = ThisBar;
   odliczanie=odliczanie+1;
 
 
   MqlRates bar1bb[];
ArraySetAsSeries(bar1bb,true);
CopyRates(symbol1,PERIOD_M5,0,7,bar1bb); 
   
  



 
   position.Select(symbol1);
   Print(bar1bb[1].open,"|>|",bar1bb[1].close,"|>|",position.PriceOpen(),"||||",profit_pips());
   if((bar1bb[1].open>=bar1bb[1].close) && (bar1bb[1].close>position.PriceOpen()) && (profit_pips()>0))
   {
   Print("buy end");
   trade.PositionClose(symbol1);
   odliczanie=0;
  
   }
   if(odliczanie>=7)
   {
   if(bar1bb[1].close<=bar1bb[1].open || bar1bb[2].close<=bar1bb[2].open || bar1bb[1].close<=bar1bb[1].open)
   {
   trade.PositionClose(symbol1);
   odliczanie=0;
   }
   }
   }
 
 }
 
 
 
 
 


    }  
  
 
   }

