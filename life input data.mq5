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

#include <Trade\Trade.mqh>
CTrade          trade;
#include <Trade\PositionInfo.mqh>
CPositionInfo position;
  


string filehandle;
string symbol1="DE30";


string data;
datetime czas;

datetime ThisBar;
static datetime LastBar = 0;
int genezis=0;
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

   
}




 void OnTick()
{
ThisBar = (datetime)SeriesInfoInteger(symbol1,PERIOD_M1,SERIES_LASTBAR_DATE);
if(LastBar != ThisBar)
  {
   printf("New bar: %s",TimeToString(ThisBar));
   LastBar = ThisBar;



 MqlRates barD1[];
ArraySetAsSeries(barD1,true);
CopyRates(symbol1,PERIOD_D1,0,4,barD1);
  

 MqlRates bar1[];
ArraySetAsSeries(bar1,true);
CopyRates(symbol1,PERIOD_M1,0,50,bar1);




czas=(bar1[1].time-(barD1[0].time));



xn0=round((double(czas)/86400)*1000000)/1000000;




//1----------------------------
 sa1=bar1[1].close;
 sa2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[1].low,bar1[2].low),bar1[3].low),bar1[4].low),bar1[5].low),bar1[6].low),bar1[7].low),bar1[8].low),bar1[9].low),bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low)-0.01;
 sa3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[1].high,bar1[2].high),bar1[3].high),bar1[4].high),bar1[5].high),bar1[6].high),bar1[7].high),bar1[8].high),bar1[9].high),bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high);



 k1=NormalizeDouble(((sa1-sa2)/(sa3-sa2)),4);
//2------------------------------
 sb1=bar1[2].close;
 sb2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[2].low,bar1[3].low),bar1[4].low),bar1[5].low),bar1[6].low),bar1[7].low),bar1[8].low),bar1[9].low),bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low)-0.01;
 sb3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[2].high,bar1[3].high),bar1[4].high),bar1[5].high),bar1[6].high),bar1[7].high),bar1[8].high),bar1[9].high),bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high);



 k2=NormalizeDouble(((sb1-sb2)/(sb3-sb2)),4);

//3-----------------------------------

 sc1=bar1[3].close;
 sc2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[3].low,bar1[4].low),bar1[5].low),bar1[6].low),bar1[7].low),bar1[8].low),bar1[9].low),bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low)-0.01;
 sc3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[3].high,bar1[4].high),bar1[5].high),bar1[6].high),bar1[7].high),bar1[8].high),bar1[9].high),bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high);



 k3=NormalizeDouble(((sc1-sc2)/(sc3-sc2)),4);

//4-----------------------------------

 sd1=bar1[4].close;
 sd2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[4].low,bar1[5].low),bar1[6].low),bar1[7].low),bar1[8].low),bar1[9].low),bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low)-0.01;
 sd3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[4].high,bar1[5].high),bar1[6].high),bar1[7].high),bar1[8].high),bar1[9].high),bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high);



 k4=NormalizeDouble(((sd1-sd2)/(sd3-sd2)),4);
//5-----------------------------------

 se1=bar1[5].close;
 se2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[5].low,bar1[6].low),bar1[7].low),bar1[8].low),bar1[9].low),bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low)-0.01;
 se3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[5].high,bar1[6].high),bar1[7].high),bar1[8].high),bar1[9].high),bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high);



 k5=NormalizeDouble(((se1-se2)/(se3-se2)),4);
//6-----------------------------------

 sf1=bar1[6].close;
 sf2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[6].low,bar1[7].low),bar1[8].low),bar1[9].low),bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low)-0.01;
 sf3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[6].high,bar1[7].high),bar1[8].high),bar1[9].high),bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high);



 k6=NormalizeDouble(((sf1-sf2)/(sf3-sf2)),4);
//7-----------------------------------

 sg1=bar1[7].close;
 sg2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[7].low,bar1[8].low),bar1[9].low),bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low)-0.01;
 sg3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[7].high,bar1[8].high),bar1[9].high),bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high);



 k7=NormalizeDouble(((sg1-sg2)/(sg3-sg2)),4);

//8-----------------------------------

 sh1=bar1[8].close;
 sh2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[8].low,bar1[9].low),bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low)-0.01;
 sh3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[8].high,bar1[9].high),bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high);



 k8=NormalizeDouble(((sh1-sh2)/(sh3-sh2)),4);

//9-----------------------------------

 si1=bar1[9].close;
 si2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[9].low,bar1[10].low),bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low)-0.01;
 si3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[9].high,bar1[10].high),bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high);



 k9=NormalizeDouble(((si1-si2)/(si3-si2)),4);

//10-----------------------------------

 sj1=bar1[10].close;
 sj2=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[10].low,bar1[11].low),bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low)-0.01;
 sj3=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[10].high,bar1[11].high),bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high);



 k10=NormalizeDouble(((sj1-sj2)/(sj3-sj2)),4);
//11------------------------------
 s111=bar1[11].close;
 s112=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[11].low,bar1[12].low),bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low),bar1[25].low)-0.01;
 s113=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[11].high,bar1[12].high),bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high),bar1[25].high);



 k11=NormalizeDouble(((s111-s112)/(s113-s112)),4);

//12-----------------------------------------
 s121=bar1[12].close;
 s122=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[12].low,bar1[13].low),bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low),bar1[25].low),bar1[26].low)-0.01;
 s123=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[12].high,bar1[13].high),bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high),bar1[25].high),bar1[26].high);



 k12=NormalizeDouble(((s121-s122)/(s123-s122)),4);
//13--------------------------------------

 s131=bar1[13].close;
 s132=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[13].low,bar1[14].low),bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low),bar1[25].low),bar1[26].low),bar1[27].low)-0.01;
 s133=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[13].high,bar1[14].high),bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high),bar1[25].high),bar1[26].high),bar1[27].high);



 k13=NormalizeDouble(((s131-s132)/(s133-s132)),4);

//14-----------------------------------
 s141=bar1[14].close;
 s142=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[14].low,bar1[15].low),bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low),bar1[25].low),bar1[26].low),bar1[27].low),bar1[28].low)-0.01;
 s143=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[14].high,bar1[15].high),bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high),bar1[25].high),bar1[26].high),bar1[27].high),bar1[28].high);



 k14=NormalizeDouble(((s141-s142)/(s143-s142)),4);
//15---------------------------------
 s151=bar1[15].close;
 s152=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[15].low,bar1[16].low),bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low),bar1[25].low),bar1[26].low),bar1[27].low),bar1[28].low),bar1[29].low)-0.01;
 s153=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[15].high,bar1[16].high),bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high),bar1[25].high),bar1[26].high),bar1[27].high),bar1[28].high),bar1[29].high);



 k15=NormalizeDouble(((s151-s152)/(s153-s152)),4);
//16---------------------------------------
 s161=bar1[16].close;
 s162=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[16].low,bar1[17].low),bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low),bar1[25].low),bar1[26].low),bar1[27].low),bar1[28].low),bar1[29].low),bar1[30].low)-0.01;
 s163=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[16].high,bar1[17].high),bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high),bar1[25].high),bar1[26].high),bar1[27].high),bar1[28].high),bar1[29].high),bar1[30].high);



 k16=NormalizeDouble(((s161-s162)/(s163-s162)),4);
//17------------------------------------
 s171=bar1[17].close;
 s172=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[17].low,bar1[18].low),bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low),bar1[25].low),bar1[26].low),bar1[27].low),bar1[28].low),bar1[29].low),bar1[30].low),bar1[31].low)-0.01;
 s173=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[17].high,bar1[18].high),bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high),bar1[25].high),bar1[26].high),bar1[27].high),bar1[28].high),bar1[29].high),bar1[30].high),bar1[31].high);



 k17=NormalizeDouble(((s171-s172)/(s173-s172)),4);
//18-----------------------------------
 s181=bar1[18].close;
 s182=MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(MathMin(bar1[18].low,bar1[19].low),bar1[20].low),bar1[21].low),bar1[22].low),bar1[23].low),bar1[24].low),bar1[25].low),bar1[26].low),bar1[27].low),bar1[28].low),bar1[29].low),bar1[30].low),bar1[31].low),bar1[32].low)-0.01;
 s183=MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(MathMax(bar1[18].high,bar1[19].high),bar1[20].high),bar1[21].high),bar1[22].high),bar1[23].high),bar1[24].high),bar1[25].high),bar1[26].high),bar1[27].high),bar1[28].high),bar1[29].high),bar1[30].high),bar1[31].high),bar1[32].high);



 k18=NormalizeDouble(((s181-s182)/(s183-s182)),4);


 x1 = NormalizeDouble((bar1[1].close-bar1[1].open)*1000,5);
 x2 = NormalizeDouble((bar1[2].close-bar1[2].open)*1000,5);
 x3 = NormalizeDouble((bar1[3].close-bar1[3].open)*1000,5);
 x4 = NormalizeDouble((bar1[4].close-bar1[4].open)*1000,5);
 x5 = NormalizeDouble((bar1[5].close-bar1[5].open)*1000,5);
 x6 = NormalizeDouble((bar1[6].close-bar1[6].open)*1000,5);
 x7 = NormalizeDouble((bar1[7].close-bar1[7].open)*1000,5);
 x8 = NormalizeDouble((bar1[8].close-bar1[8].open)*1000,5);
 x9 = NormalizeDouble((bar1[9].close-bar1[9].open)*1000,5);
 x10 = NormalizeDouble((bar1[10].close-bar1[10].open)*1000,5);
 x11 = NormalizeDouble((bar1[11].close-bar1[11].open)*1000,5);
 x12 = NormalizeDouble((bar1[12].close-bar1[12].open)*1000,5);
 x13 = NormalizeDouble((bar1[13].close-bar1[13].open)*1000,5);
 x14 = NormalizeDouble((bar1[14].close-bar1[14].open)*1000,5);
 x15 = NormalizeDouble((bar1[15].close-bar1[15].open)*1000,5);
 x16 = NormalizeDouble((bar1[16].close-bar1[16].open)*1000,5);
 x17 = NormalizeDouble((bar1[17].close-bar1[17].open)*1000,5);
 x18 = NormalizeDouble((bar1[18].close-bar1[18].open)*1000,5);



//1---------------------------------
if (bar1[1].close>=bar1[1].open)
{
y1 = NormalizeDouble((bar1[1].high-bar1[1].close)*100,5);
z1 = NormalizeDouble((bar1[1].low-bar1[1].open)*100,5);
}
if (bar1[1].open>bar1[1].close)
{
y1 = NormalizeDouble((bar1[1].high-bar1[1].open)*100,5);
z1 = NormalizeDouble((bar1[1].low-bar1[1].close)*100,5);
}
//2---------------------------------
if (bar1[2].close>=bar1[2].open)
{
y2 = NormalizeDouble((bar1[2].high-bar1[2].close)*100,5);
z2 = NormalizeDouble((bar1[2].low-bar1[2].open)*100,5);
}
if (bar1[2].open>bar1[2].close)
{
y2 = NormalizeDouble((bar1[2].high-bar1[2].open)*100,5);
z2 = NormalizeDouble((bar1[2].low-bar1[2].close)*100,5);
}
//3---------------------------------
if (bar1[3].close>=bar1[3].open)
{
y3 = NormalizeDouble((bar1[3].high-bar1[3].close)*100,5);
z3 = NormalizeDouble((bar1[3].low-bar1[3].open)*100,5);
}
if (bar1[3].open>bar1[3].close)
{
y3 = NormalizeDouble((bar1[3].high-bar1[3].open)*100,5);
z3 = NormalizeDouble((bar1[3].low-bar1[3].close)*100,5);
}
//4---------------------------------
if (bar1[4].close>=bar1[4].open)
{
y4 = NormalizeDouble((bar1[4].high-bar1[4].close)*100,5);
z4 = NormalizeDouble((bar1[4].low-bar1[4].open)*100,5);
}
if (bar1[4].open>bar1[4].close)
{
y4 = NormalizeDouble((bar1[4].high-bar1[4].open)*100,5);
z4 = NormalizeDouble((bar1[4].low-bar1[4].close)*100,5);
}
//5---------------------------------
if (bar1[5].close>=bar1[5].open)
{
y5 = NormalizeDouble((bar1[5].high-bar1[5].close)*100,5);
z5 = NormalizeDouble((bar1[5].low-bar1[5].open)*100,5);
}
if (bar1[5].open>bar1[5].close)
{
y5 = NormalizeDouble((bar1[5].high-bar1[5].open)*100,5);
z5 = NormalizeDouble((bar1[5].low-bar1[5].close)*100,5);
}
//6---------------------------------
if (bar1[6].close>=bar1[6].open)
{
y6 = NormalizeDouble((bar1[6].high-bar1[6].close)*100,5);
z6 = NormalizeDouble((bar1[6].low-bar1[6].open)*100,5);
}
if (bar1[6].open>bar1[6].close)
{
y6 = NormalizeDouble((bar1[6].high-bar1[6].open)*100,5);
z6 = NormalizeDouble((bar1[6].low-bar1[6].close)*100,5);
}
//7---------------------------------
if (bar1[7].close>=bar1[7].open)
{
y7 = NormalizeDouble((bar1[7].high-bar1[7].close)*100,5);
z7 = NormalizeDouble((bar1[7].low-bar1[7].open)*100,5);
}
if (bar1[7].open>bar1[7].close)
{
y7 = NormalizeDouble((bar1[7].high-bar1[7].open)*100,5);
z7 = NormalizeDouble((bar1[7].low-bar1[7].close)*100,5);
}
//8---------------------------------
if (bar1[8].close>=bar1[8].open)
{
y8 = NormalizeDouble((bar1[8].high-bar1[8].close)*100,5);
z8 = NormalizeDouble((bar1[8].low-bar1[8].open)*100,5);
}
if (bar1[8].open>bar1[8].close)
{
y8 = NormalizeDouble((bar1[8].high-bar1[8].open)*100,5);
z8 = NormalizeDouble((bar1[8].low-bar1[8].close)*100,5);
}
//9---------------------------------
if (bar1[9].close>=bar1[9].open)
{
y9 = NormalizeDouble((bar1[9].high-bar1[9].close)*100,5);
z9 = NormalizeDouble((bar1[9].low-bar1[9].open)*100,5);
}
if (bar1[9].open>bar1[9].close)
{
y9 = NormalizeDouble((bar1[9].high-bar1[9].open)*100,5);
z9 = NormalizeDouble((bar1[9].low-bar1[9].close)*100,5);
}
//10---------------------------------
if (bar1[10].close>=bar1[10].open)
{
y10 = NormalizeDouble((bar1[10].high-bar1[10].close)*100,5);
z10 = NormalizeDouble((bar1[10].low-bar1[10].open)*100,5);
}
if (bar1[10].open>bar1[10].close)
{
y10 = NormalizeDouble((bar1[10].high-bar1[10].open)*100,5);
z10 = NormalizeDouble((bar1[10].low-bar1[10].close)*100,5);
}
//11---------------------------------
if (bar1[11].close>=bar1[11].open)
{
y11 = NormalizeDouble((bar1[11].high-bar1[11].close)*100,5);
z11 = NormalizeDouble((bar1[11].low-bar1[11].open)*100,5);
}
if (bar1[11].open>bar1[11].close)
{
y11 = NormalizeDouble((bar1[11].high-bar1[11].open)*100,5);
z11 = NormalizeDouble((bar1[11].low-bar1[11].close)*100,5);
}
//12---------------------------------
if (bar1[12].close>=bar1[12].open)
{
y12 = NormalizeDouble((bar1[12].high-bar1[12].close)*100,5);
z12 = NormalizeDouble((bar1[12].low-bar1[12].open)*100,5);
}
if (bar1[12].open>bar1[12].close)
{
y12 = NormalizeDouble((bar1[12].high-bar1[12].open)*100,5);
z12 = NormalizeDouble((bar1[12].low-bar1[12].close)*100,5);
}
//1---------------------------------
if (bar1[13].close>=bar1[13].open)
{
y13 = NormalizeDouble((bar1[13].high-bar1[13].close)*100,5);
z13 = NormalizeDouble((bar1[13].low-bar1[13].open)*100,5);
}
if (bar1[13].open>bar1[13].close)
{
y13 = NormalizeDouble((bar1[13].high-bar1[13].open)*100,5);
z13 = NormalizeDouble((bar1[13].low-bar1[13].close)*100,5);
}
//14---------------------------------
if (bar1[14].close>=bar1[14].open)
{
y14 = NormalizeDouble((bar1[14].high-bar1[14].close)*100,5);
z14 = NormalizeDouble((bar1[14].low-bar1[14].open)*100,5);
}
if (bar1[14].open>bar1[14].close)
{
y14 = NormalizeDouble((bar1[14].high-bar1[14].open)*100,5);
z14 = NormalizeDouble((bar1[14].low-bar1[14].close)*100,5);
}
//15---------------------------------
if (bar1[15].close>=bar1[15].open)
{
y15 = NormalizeDouble((bar1[15].high-bar1[15].close)*100,5);
z15 = NormalizeDouble((bar1[15].low-bar1[15].open)*100,5);
}
if (bar1[15].open>bar1[15].close)
{
y15 = NormalizeDouble((bar1[15].high-bar1[15].open)*100,5);
z15 = NormalizeDouble((bar1[15].low-bar1[15].close)*100,5);
}
//16---------------------------------
if (bar1[16].close>=bar1[16].open)
{
y16 = NormalizeDouble((bar1[16].high-bar1[16].close)*100,5);
z16 = NormalizeDouble((bar1[16].low-bar1[16].open)*100,5);
}
if (bar1[16].open>bar1[16].close)
{
y16 = NormalizeDouble((bar1[16].high-bar1[16].open)*100,5);
z16 = NormalizeDouble((bar1[16].low-bar1[16].close)*100,5);
}
//17---------------------------------
if (bar1[17].close>=bar1[17].open)
{
y17 = NormalizeDouble((bar1[17].high-bar1[17].close)*100,5);
z17 = NormalizeDouble((bar1[17].low-bar1[17].open)*100,5);
}
if (bar1[17].open>bar1[17].close)
{
y17 = NormalizeDouble((bar1[17].high-bar1[17].open)*100,5);
z17 = NormalizeDouble((bar1[17].low-bar1[17].close)*100,5);
}
//18---------------------------------
if (bar1[18].close>=bar1[18].open)
{
y18 = NormalizeDouble((bar1[18].high-bar1[18].close)*100,5);
z18 = NormalizeDouble((bar1[18].low-bar1[18].open)*100,5);
}
if (bar1[18].open>bar1[18].close)
{
y18 = NormalizeDouble((bar1[18].high-bar1[18].open)*100,5);
z18 = NormalizeDouble((bar1[18].low-bar1[18].close)*100,5);
}




if((0.39583333333333 < xn0) && (xn0  < 0.7916666666666))
{

data = k18+","+x18+","+y18+","+z18+","+k17+","+x17+","+y17+","+z17+","+k16+","+x16+","+y16+","+z16+","+
k15+","+x15+","+y15+","+z15+","+k14+","+x14+","+y14+","+z14+","+k13+","+x13+","+y13+","+z13+","+
k12+","+x12+","+y12+","+z12+","+k11+","+x11+","+y11+","+z11+","+k10+","+x10+","+y10+","+z10+","+
k9+","+x9+","+y9+","+z9+","+k8+","+x8+","+y8+","+z8+","+k7+","+x7+","+y7+","+z7+","+
k6+","+x6+","+y6+","+z6+","+k5+","+x5+","+y5+","+z5+","+k4+","+x4+","+y4+","+z4+","+
k3+","+x3+","+y3+","+z3+","+k2+","+x2+","+y2+","+z2+","+k1+","+x1+","+y1+","+z1;
   
filehandle=FileOpen("live data.csv",FILE_WRITE|FILE_CSV,',');

FileWriteString(filehandle, data);
      
FileClose(filehandle);
}
}

}