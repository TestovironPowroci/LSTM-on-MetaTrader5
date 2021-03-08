//+------------------------------------------------------------------+
//|                                                     the end .mq5 |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2018, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
#include <Trade\SymbolInfo.mqh>  
CSymbolInfo    m_symbol;  
#include <Trade\Trade.mqh>
CTrade          trade;


string filehandle;
string symbol1="DE30";


double bk;
int dnc15;
string data;
datetime czas;
double xn0;

double k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25,k26,k27,k28,k29,k30,k31,k32,k33,k34,k35,k36,k37,k38,k39,k40,k41,k42,k43,k44,k45,k46,k47,k48,k49,k50,k51,k52,k53,k54,k55,k56,k57,k58,k59,k60;
double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49,x50,x51,x52,x53,x54,x55,x56,x57,x58,x59,x60;
double y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27,y28,y29,y30,y31,y32,y33,y34,y35,y36,y37,y38,y39,y40,y41,y42,y43,y44,y45,y46,y47,y48,y49,y50,y51,y52,y53,y54,y55,y56,y57,y58,y59,y60;
double z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25,z26,z27,z28,z29,z30,z31,z32,z33,z34,z35,z36,z37,z38,z39,z40,z41,z42,z43,z44,z45,z46,z47,z48,z49,z50,z51,z52,z53,z54,z55,z56,z57,z58,z59,z60;
double sa1,sb1,sc1,sd1,se1,sf1,sg1,sh1,si1,sj1,s111,s121,s131,s141,s151,s161,s171,s181,s191,s201,s211,s221,s231,s241,s251,s261,s271,s281,s291,s301,s311,s321,s331,s341,s351,s361,s371,s381,s391,s401,s411,s421,s431,s441,s451,s461,s471,s481,s491,s501,s511,s521,s531,s541,s551,s561,s571,s581,s591,s601;
double sa2,sb2,sc2,sd2,se2,sf2,sg2,sh2,si2,sj2,s112,s122,s132,s142,s152,s162,s172,s182,s192,s202,s212,s222,s232,s242,s252,s262,s272,s282,s292,s302,s312,s322,s332,s342,s352,s362,s372,s382,s392,s402,s412,s422,s432,s442,s452,s462,s472,s482,s492,s502,s512,s522,s532,s542,s552,s562,s572,s582,s592,s602;
double sa3,sb3,sc3,sd3,se3,sf3,sg3,sh3,si3,sj3,s113,s123,s133,s143,s153,s163,s173,s183,s193,s203,s213,s223,s233,s243,s253,s263,s273,s283,s293,s303,s313,s323,s333,s343,s353,s363,s373,s383,s393,s403,s413,s423,s433,s443,s453,s463,s473,s483,s493,s503,s513,s523,s533,s543,s553,s563,s573,s583,s593,s603;



void  OnInit()

{
ResetLastError();

filehandle=FileOpen("learning dax30.csv",FILE_WRITE|FILE_CSV,',');

   if(filehandle!=INVALID_HANDLE)
     {      
      Print("File opened correctly");
     }
   else Print("Error in opening file,",GetLastError());

   return;
}



 void OnTick()
 {
 
 

 MqlRates bar1[];
ArraySetAsSeries(bar1,true);
CopyRates(symbol1,PERIOD_M1,0,62,bar1);

 MqlRates barD1[];
ArraySetAsSeries(bar1,true);
CopyRates(symbol1,PERIOD_D1,0,2,bar1);

xn0=round((double(czas)/86400)*1000000)/1000000;
czas=(bar1[30].time-(barD1[0].time));





//1----------------------------
 sa1=bar1[31].close;
 sa2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[31].low,bar1[32].low),bar1[33].low),bar1[34].low),bar1[35].low),bar1[36].low),bar1[37].low),bar1[38].low),bar1[39].low),bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low)-0.01;
 sa3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[31].high,bar1[32].high),bar1[33].high),bar1[34].high),bar1[35].high),bar1[36].high),bar1[37].high),bar1[38].high),bar1[39].high),bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high);



 k1=NormalizeDouble(((sa1-sa2)/(sa3-sa2)),4);
//2------------------------------
 sb1=bar1[32].close;
 sb2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[32].low,bar1[33].low),bar1[34].low),bar1[35].low),bar1[36].low),bar1[37].low),bar1[38].low),bar1[39].low),bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low)-0.01;
 sb3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[32].high,bar1[33].high),bar1[34].high),bar1[35].high),bar1[36].high),bar1[37].high),bar1[38].high),bar1[39].high),bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high);



 k2=NormalizeDouble(((sb1-sb2)/(sb3-sb2)),4);

//3-----------------------------------

 sc1=bar1[33].close;
 sc2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[33].low,bar1[34].low),bar1[35].low),bar1[36].low),bar1[37].low),bar1[38].low),bar1[39].low),bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low)-0.01;
 sc3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[33].high,bar1[34].high),bar1[35].high),bar1[36].high),bar1[37].high),bar1[38].high),bar1[39].high),bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high);



 k3=NormalizeDouble(((sc1-sc2)/(sc3-sc2)),4);

//4-----------------------------------

 sd1=bar1[34].close;
 sd2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[34].low,bar1[35].low),bar1[36].low),bar1[37].low),bar1[38].low),bar1[39].low),bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low)-0.01;
 sd3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[34].high,bar1[35].high),bar1[36].high),bar1[37].high),bar1[38].high),bar1[39].high),bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high);



 k4=NormalizeDouble(((sd1-sd2)/(sd3-sd2)),4);
//5-----------------------------------

 se1=bar1[35].close;
 se2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[35].low,bar1[36].low),bar1[37].low),bar1[38].low),bar1[39].low),bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low)-0.01;
 se3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[35].high,bar1[36].high),bar1[37].high),bar1[38].high),bar1[39].high),bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high);



 k5=NormalizeDouble(((se1-se2)/(se3-se2)),4);
//6-----------------------------------

 sf1=bar1[36].close;
 sf2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[36].low,bar1[37].low),bar1[38].low),bar1[39].low),bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low)-0.01;
 sf3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[36].high,bar1[37].high),bar1[38].high),bar1[39].high),bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high);



 k6=NormalizeDouble(((sf1-sf2)/(sf3-sf2)),4);
//7-----------------------------------

 sg1=bar1[37].close;
 sg2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[37].low,bar1[38].low),bar1[39].low),bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low)-0.01;
 sg3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[37].high,bar1[38].high),bar1[39].high),bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high);



 k7=NormalizeDouble(((sg1-sg2)/(sg3-sg2)),4);

//8-----------------------------------

 sh1=bar1[38].close;
 sh2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[38].low,bar1[39].low),bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low)-0.01;
 sh3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[38].high,bar1[39].high),bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high);



 k8=NormalizeDouble(((sh1-sh2)/(sh3-sh2)),4);

//9-----------------------------------

 si1=bar1[39].close;
 si2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[39].low,bar1[40].low),bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low)-0.01;
 si3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[39].high,bar1[40].high),bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high);



 k9=NormalizeDouble(((si1-si2)/(si3-si2)),4);

//10-----------------------------------

 sj1=bar1[40].close;
 sj2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[40].low,bar1[41].low),bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low)-0.01;
 sj3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[40].high,bar1[41].high),bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high);



 k10=NormalizeDouble(((sj1-sj2)/(sj3-sj2)),4);
//11------------------------------
 s111=bar1[41].close;
 s112=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[41].low,bar1[42].low),bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low),bar1[55].low)-0.01;
 s113=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[41].high,bar1[42].high),bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high),bar1[55].high);



 k11=NormalizeDouble(((s111-s112)/(s113-s112)),4);

//12-----------------------------------------
 s121=bar1[42].close;
 s122=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[42].low,bar1[43].low),bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low),bar1[55].low),bar1[56].low)-0.01;
 s123=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[42].high,bar1[43].high),bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high),bar1[55].high),bar1[56].high);



 k12=NormalizeDouble(((s121-s122)/(s123-s122)),4);
//13--------------------------------------

 s131=bar1[43].close;
 s132=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[43].low,bar1[44].low),bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low),bar1[55].low),bar1[56].low),bar1[57].low)-0.01;
 s133=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[43].high,bar1[44].high),bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high),bar1[55].high),bar1[56].high),bar1[57].high);



 k13=NormalizeDouble(((s131-s132)/(s133-s132)),4);

//14-----------------------------------
 s141=bar1[44].close;
 s142=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[44].low,bar1[45].low),bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low),bar1[55].low),bar1[56].low),bar1[57].low),bar1[58].low)-0.01;
 s143=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[44].high,bar1[45].high),bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high),bar1[55].high),bar1[56].high),bar1[57].high),bar1[58].high);



 k14=NormalizeDouble(((s141-s142)/(s143-s142)),4);
//15---------------------------------
 s151=bar1[45].close;
 s152=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[45].low,bar1[46].low),bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low),bar1[55].low),bar1[56].low),bar1[57].low),bar1[58].low),bar1[59].low)-0.01;
 s153=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[45].high,bar1[46].high),bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high),bar1[55].high),bar1[56].high),bar1[57].high),bar1[58].high),bar1[59].high);



 k15=NormalizeDouble(((s151-s152)/(s153-s152)),4);
//16---------------------------------------
 s161=bar1[46].close;
 s162=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[46].low,bar1[47].low),bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low),bar1[55].low),bar1[56].low),bar1[57].low),bar1[58].low),bar1[59].low),bar1[60].low)-0.01;
 s163=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[46].high,bar1[47].high),bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high),bar1[55].high),bar1[56].high),bar1[57].high),bar1[58].high),bar1[59].high),bar1[60].high);



 k16=NormalizeDouble(((s161-s162)/(s163-s162)),4);
//17------------------------------------
 s171=bar1[47].close;
 s172=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[47].low,bar1[48].low),bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low),bar1[55].low),bar1[56].low),bar1[57].low),bar1[58].low),bar1[59].low),bar1[60].low),bar1[61].low)-0.01;
 s173=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[47].high,bar1[48].high),bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high),bar1[55].high),bar1[56].high),bar1[57].high),bar1[58].high),bar1[59].high),bar1[60].high),bar1[61].high);



 k17=NormalizeDouble(((s171-s172)/(s173-s172)),4);
//18-----------------------------------
 s181=bar1[48].close;
 s182=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[48].low,bar1[49].low),bar1[50].low),bar1[51].low),bar1[52].low),bar1[53].low),bar1[54].low),bar1[55].low),bar1[56].low),bar1[57].low),bar1[58].low),bar1[59].low),bar1[60].low),bar1[61].low),bar1[62].low)-0.01;
 s183=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[48].high,bar1[49].high),bar1[50].high),bar1[51].high),bar1[52].high),bar1[53].high),bar1[54].high),bar1[55].high),bar1[56].high),bar1[57].high),bar1[58].high),bar1[59].high),bar1[60].high),bar1[61].high),bar1[62].high);



 k18=NormalizeDouble(((s181-s182)/(s183-s182)),4);





 x1 = NormalizeDouble((bar1[31].close-bar1[31].open)*1000,5);
 x2 = NormalizeDouble((bar1[32].close-bar1[32].open)*1000,5);
 x3 = NormalizeDouble((bar1[33].close-bar1[33].open)*1000,5);
 x4 = NormalizeDouble((bar1[34].close-bar1[34].open)*1000,5);
 x5 = NormalizeDouble((bar1[35].close-bar1[35].open)*1000,5);
 x6 = NormalizeDouble((bar1[36].close-bar1[36].open)*1000,5);
 x7 = NormalizeDouble((bar1[37].close-bar1[37].open)*1000,5);
 x8 = NormalizeDouble((bar1[38].close-bar1[38].open)*1000,5);
 x9 = NormalizeDouble((bar1[39].close-bar1[39].open)*1000,5);
 x10 = NormalizeDouble((bar1[40].close-bar1[40].open)*1000,5);
 x11 = NormalizeDouble((bar1[41].close-bar1[41].open)*1000,5);
 x12 = NormalizeDouble((bar1[42].close-bar1[42].open)*1000,5);
 x13 = NormalizeDouble((bar1[43].close-bar1[43].open)*1000,5);
 x14 = NormalizeDouble((bar1[44].close-bar1[44].open)*1000,5);
 x15 = NormalizeDouble((bar1[45].close-bar1[45].open)*1000,5);
 x16 = NormalizeDouble((bar1[46].close-bar1[46].open)*1000,5);
 x17 = NormalizeDouble((bar1[47].close-bar1[47].open)*1000,5);
 x18 = NormalizeDouble((bar1[48].close-bar1[48].open)*1000,5);



//1---------------------------------
if (bar1[31].close>=bar1[31].open)
{
y1 = NormalizeDouble((bar1[31].high-bar1[31].close)*100,5);
z1 = NormalizeDouble((bar1[31].low-bar1[31].open)*100,5);
}
if (bar1[31].open>bar1[31].close)
{
y1 = NormalizeDouble((bar1[31].high-bar1[31].open)*100,5);
z1 = NormalizeDouble((bar1[31].low-bar1[31].close)*100,5);
}
//2---------------------------------
if (bar1[32].close>=bar1[32].open)
{
y2 = NormalizeDouble((bar1[32].high-bar1[32].close)*100,5);
z2 = NormalizeDouble((bar1[32].low-bar1[32].open)*100,5);
}
if (bar1[32].open>bar1[32].close)
{
y2 = NormalizeDouble((bar1[32].high-bar1[32].open)*100,5);
z2 = NormalizeDouble((bar1[32].low-bar1[32].close)*100,5);
}
//3---------------------------------
if (bar1[33].close>=bar1[33].open)
{
y3 = NormalizeDouble((bar1[33].high-bar1[33].close)*100,5);
z3 = NormalizeDouble((bar1[33].low-bar1[33].open)*100,5);
}
if (bar1[33].open>bar1[33].close)
{
y3 = NormalizeDouble((bar1[33].high-bar1[33].open)*100,5);
z3 = NormalizeDouble((bar1[33].low-bar1[33].close)*100,5);
}
//4---------------------------------
if (bar1[34].close>=bar1[34].open)
{
y4 = NormalizeDouble((bar1[34].high-bar1[34].close)*100,5);
z4 = NormalizeDouble((bar1[34].low-bar1[34].open)*100,5);
}
if (bar1[34].open>bar1[34].close)
{
y4 = NormalizeDouble((bar1[34].high-bar1[34].open)*100,5);
z4 = NormalizeDouble((bar1[34].low-bar1[34].close)*100,5);
}
//5---------------------------------
if (bar1[35].close>=bar1[35].open)
{
y5 = NormalizeDouble((bar1[35].high-bar1[35].close)*100,5);
z5 = NormalizeDouble((bar1[35].low-bar1[35].open)*100,5);
}
if (bar1[35].open>bar1[35].close)
{
y5 = NormalizeDouble((bar1[35].high-bar1[35].open)*100,5);
z5 = NormalizeDouble((bar1[35].low-bar1[35].close)*100,5);
}
//6---------------------------------
if (bar1[36].close>=bar1[36].open)
{
y6 = NormalizeDouble((bar1[36].high-bar1[36].close)*100,5);
z6 = NormalizeDouble((bar1[36].low-bar1[36].open)*100,5);
}
if (bar1[36].open>bar1[36].close)
{
y6 = NormalizeDouble((bar1[36].high-bar1[36].open)*100,5);
z6 = NormalizeDouble((bar1[36].low-bar1[36].close)*100,5);
}
//7---------------------------------
if (bar1[37].close>=bar1[37].open)
{
y7 = NormalizeDouble((bar1[37].high-bar1[37].close)*100,5);
z7 = NormalizeDouble((bar1[37].low-bar1[37].open)*100,5);
}
if (bar1[37].open>bar1[37].close)
{
y7 = NormalizeDouble((bar1[37].high-bar1[37].open)*100,5);
z7 = NormalizeDouble((bar1[37].low-bar1[37].close)*100,5);
}
//8---------------------------------
if (bar1[38].close>=bar1[38].open)
{
y8 = NormalizeDouble((bar1[38].high-bar1[38].close)*100,5);
z8 = NormalizeDouble((bar1[38].low-bar1[38].open)*100,5);
}
if (bar1[38].open>bar1[38].close)
{
y8 = NormalizeDouble((bar1[38].high-bar1[38].open)*100,5);
z8 = NormalizeDouble((bar1[38].low-bar1[38].close)*100,5);
}
//9---------------------------------
if (bar1[39].close>=bar1[39].open)
{
y9 = NormalizeDouble((bar1[39].high-bar1[39].close)*100,5);
z9 = NormalizeDouble((bar1[39].low-bar1[39].open)*100,5);
}
if (bar1[39].open>bar1[39].close)
{
y9 = NormalizeDouble((bar1[39].high-bar1[39].open)*100,5);
z9 = NormalizeDouble((bar1[39].low-bar1[39].close)*100,5);
}
//10---------------------------------
if (bar1[40].close>=bar1[40].open)
{
y10 = NormalizeDouble((bar1[40].high-bar1[40].close)*100,5);
z10 = NormalizeDouble((bar1[40].low-bar1[40].open)*100,5);
}
if (bar1[40].open>bar1[40].close)
{
y10 = NormalizeDouble((bar1[40].high-bar1[40].open)*100,5);
z10 = NormalizeDouble((bar1[40].low-bar1[40].close)*100,5);
}
//11---------------------------------
if (bar1[41].close>=bar1[41].open)
{
y11 = NormalizeDouble((bar1[41].high-bar1[41].close)*100,5);
z11 = NormalizeDouble((bar1[41].low-bar1[41].open)*100,5);
}
if (bar1[41].open>bar1[41].close)
{
y11 = NormalizeDouble((bar1[41].high-bar1[41].open)*100,5);
z11 = NormalizeDouble((bar1[41].low-bar1[41].close)*100,5);
}
//12---------------------------------
if (bar1[42].close>=bar1[42].open)
{
y12 = NormalizeDouble((bar1[42].high-bar1[42].close)*100,5);
z12 = NormalizeDouble((bar1[42].low-bar1[42].open)*100,5);
}
if (bar1[42].open>bar1[42].close)
{
y12 = NormalizeDouble((bar1[42].high-bar1[42].open)*100,5);
z12 = NormalizeDouble((bar1[42].low-bar1[42].close)*100,5);
}
//13---------------------------------
if (bar1[43].close>=bar1[43].open)
{
y13 = NormalizeDouble((bar1[43].high-bar1[43].close)*100,5);
z13 = NormalizeDouble((bar1[43].low-bar1[43].open)*100,5);
}
if (bar1[43].open>bar1[43].close)
{
y13 = NormalizeDouble((bar1[43].high-bar1[43].open)*100,5);
z13 = NormalizeDouble((bar1[43].low-bar1[43].close)*100,5);
}
//14---------------------------------
if (bar1[44].close>=bar1[44].open)
{
y14 = NormalizeDouble((bar1[44].high-bar1[44].close)*100,5);
z14 = NormalizeDouble((bar1[44].low-bar1[44].open)*100,5);
}
if (bar1[44].open>bar1[44].close)
{
y14 = NormalizeDouble((bar1[44].high-bar1[44].open)*100,5);
z14 = NormalizeDouble((bar1[44].low-bar1[44].close)*100,5);
}
//15---------------------------------
if (bar1[45].close>=bar1[45].open)
{
y15 = NormalizeDouble((bar1[45].high-bar1[45].close)*100,5);
z15 = NormalizeDouble((bar1[45].low-bar1[45].open)*100,5);
}
if (bar1[45].open>bar1[45].close)
{
y15 = NormalizeDouble((bar1[45].high-bar1[45].open)*100,5);
z15 = NormalizeDouble((bar1[45].low-bar1[45].close)*100,5);
}
//16---------------------------------
if (bar1[46].close>=bar1[46].open)
{
y16 = NormalizeDouble((bar1[46].high-bar1[46].close)*100,5);
z16 = NormalizeDouble((bar1[46].low-bar1[46].open)*100,5);
}
if (bar1[46].open>bar1[46].close)
{
y16 = NormalizeDouble((bar1[46].high-bar1[46].open)*100,5);
z16 = NormalizeDouble((bar1[46].low-bar1[46].close)*100,5);
}
//17---------------------------------
if (bar1[47].close>=bar1[47].open)
{
y17 = NormalizeDouble((bar1[47].high-bar1[47].close)*100,5);
z17 = NormalizeDouble((bar1[47].low-bar1[47].open)*100,5);
}
if (bar1[47].open>bar1[47].close)
{
y17 = NormalizeDouble((bar1[47].high-bar1[47].open)*100,5);
z17 = NormalizeDouble((bar1[47].low-bar1[47].close)*100,5);
}
//18---------------------------------
if (bar1[48].close>=bar1[48].open)
{
y18 = NormalizeDouble((bar1[48].high-bar1[48].close)*100,5);
z18 = NormalizeDouble((bar1[48].low-bar1[48].open)*100,5);
}
if (bar1[48].open>bar1[48].close)
{
y18 = NormalizeDouble((bar1[48].high-bar1[48].open)*100,5);
z18 = NormalizeDouble((bar1[48].low-bar1[48].close)*100,5);
}


dnc15 = 0;


bk=bar1[16].close-bar1[31].close;
//BUY
if (bk>=15)
{
dnc15=2;
}



//SELL
if (bk<=-15)
{
dnc15=1;
}





Sleep(60000);




if((0.39583333333333 < xn0) && (xn0  < 0.7916666666666))
{

data = k18+","+x18+","+y18+","+z18+","+k17+","+x17+","+y17+","+z17+","+k16+","+x16+","+y16+","+z16+","+
k15+","+x15+","+y15+","+z15+","+k14+","+x14+","+y14+","+z14+","+k13+","+x13+","+y13+","+z13+","+
k12+","+x12+","+y12+","+z12+","+k11+","+x11+","+y11+","+z11+","+k10+","+x10+","+y10+","+z10+","+
k9+","+x9+","+y9+","+z9+","+k8+","+x8+","+y8+","+z8+","+k7+","+x7+","+y7+","+z7+","+
k6+","+x6+","+y6+","+z6+","+k5+","+x5+","+y5+","+z5+","+k4+","+x4+","+y4+","+z4+","+
k3+","+x3+","+y3+","+z3+","+k2+","+x2+","+y2+","+z2+","+k1+","+x1+","+y1+","+z1+","+dnc15;
   
FileWriteString(filehandle, data);
}  

}

	
		

void OnDeinit(const int reason)
{
 FileClose(filehandle);
  
}