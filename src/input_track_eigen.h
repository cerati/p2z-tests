ATRK inputtrk;
const float tmptp[6] = {20.0483646, 86.4534073, -271.755951, 0.0838338137, 1.38423216, 2.82098508};
const float tmptc[21] = { 6.82491373e-05, 0.000272378646, 0.00132922246, 0, 0, 0, -1.35480389e-06,
			  -1.76664173e-06, 0, 3.10420774e-07, -6.30985767e-07, 8.69705445e-08, 0, 8.98418477e-08,
			  4.58238887e-08, -1.7055487e-06, -8.16863303e-06, 0, 1.30164102e-08, 8.22342194e-10, 6.65813857e-08};
inputtrk.par = Matrix<float,6,1>(tmptp);
inputtrk.cov = Matrix<float,21,1>(tmptc);
inputtrk.q = -1;

AHIT inputhit00;
const float tmphp00[3] = {0.870918512, 3.15948129, -14.3067408};
const float tmphc00[6] = {2.19158505e-06, -5.49770505e-07, 1.37912821e-07, 4.82739959e-10, 7.37647388e-10, 2.77211802e-05};
inputhit00.pos = Matrix<float,3,1>(tmphp00);
inputhit00.cov = Matrix<float,6,1>(tmphc00);

AHIT inputhit01;
const float tmphp01[3] = {1.74246395, 6.35200739, -24.2843418};
const float tmphc01[6] = {4.75911293e-07, -1.57801381e-07, 5.2323351e-08, -5.34601981e-11, 4.05433933e-12, 3.72799877e-06};
inputhit01.pos = Matrix<float,3,1>(tmphp01);
inputhit01.cov = Matrix<float,6,1>(tmphc01);

AHIT inputhit02;
const float tmphp02[3] = {2.65926456, 9.75921059, -34.9179306};
const float tmphc02[6] = {1.29466298e-06, 2.58482282e-06, 1.16544898e-05, -9.22727565e-07, -2.86404656e-06, 8.18422848e-07};
inputhit02.pos = Matrix<float,3,1>(tmphp02);
inputhit02.cov = Matrix<float,6,1>(tmphc02);

AHIT inputhit03;
const float tmphp03[3] = {3.01125336, 11.0806084, -39.0169144};
const float tmphc03[6] = {7.78740628e-07, 8.8761135e-07, 2.75849015e-06, -1.84013274e-07, -3.63300501e-08, 6.06964363e-08};
inputhit03.pos = Matrix<float,3,1>(tmphp03);
inputhit03.cov = Matrix<float,6,1>(tmphc03);

AHIT inputhit04;
const float tmphp04[3] = {3.02387047, 11.1359024, -39.1962624};
const float tmphc04[6] = {1.34629988e-06, 5.50397715e-07, 4.07115658e-06, -4.17057151e-07, 2.12687837e-07, 1.6737323e-07};
inputhit04.pos = Matrix<float,3,1>(tmphp04);
inputhit04.cov = Matrix<float,6,1>(tmphc04);

AHIT inputhit05;
const float tmphp05[3] = {3.82721639, 14.1817942, -48.6744347};
const float tmphc05[6] = {1.26740929e-06, 1.34525649e-07, 1.8719951e-06, -4.30450768e-07, 1.39769455e-07, 1.64708752e-07};
inputhit05.pos = Matrix<float,3,1>(tmphp05);
inputhit05.cov = Matrix<float,6,1>(tmphc05);

AHIT inputhit06;
const float tmphp06[3] = {7.32026434, 27.7226219, -80.5239563};
const float tmphc06[6] = {0.667558372, 2.52804375, 9.57383251, 3.1761835e-16, 1.20283655e-15, 1.51121895e-31};
inputhit06.pos = Matrix<float,3,1>(tmphp06);
inputhit06.cov = Matrix<float,6,1>(tmphc06);

AHIT inputhit07;
const float tmphp07[3] = {7.69805288, 27.5908051, -80.279007};
const float tmphc07[6] = {1.35411382, 3.46923542, 8.88823414, -0, 0, 0};
inputhit07.pos = Matrix<float,3,1>(tmphp07);
inputhit07.cov = Matrix<float,6,1>(tmphc07);

AHIT inputhit08;
const float tmphp08[3] = {7.2608819, 27.7381134, -93.4740219};
const float tmphc08[6] = {0.65680629, 2.50903654, 9.58479023, -3.15138714e-16, -1.20386336e-15, 1.51206953e-31};
inputhit08.pos = Matrix<float,3,1>(tmphp08);
inputhit08.cov = Matrix<float,6,1>(tmphc08);

AHIT inputhit09;
const float tmphp09[3] =  {7.20401049, 27.7534981, -106.423027};
const float tmphc09[6] = {0.646517396, 2.4906795, 9.59536648, -3.12745263e-16, -1.20485344e-15, 1.51288838e-31};
inputhit09.pos = Matrix<float,3,1>(tmphp09);
inputhit09.cov = Matrix<float,6,1>(tmphc09);

AHIT inputhit10;
const float tmphp10[3] = {9.20536804, 35.535717, -109.922653};
const float tmphc10[6] = {0.40738529, 1.57248175, 6.06989956, 1.97438165e-16, 7.62125518e-16, 9.56910882e-32};
inputhit10.pos = Matrix<float,3,1>(tmphp10);
inputhit10.cov = Matrix<float,6,1>(tmphc10);

AHIT inputhit11;
const float tmphp11[3] = {6.81385803, 27.9251652, -106.178093};
const float tmphc11[6] = {1.13409436, 3.21469259, 9.11241245, -0, 0, 0};
inputhit11.pos = Matrix<float,3,1>(tmphp11);
inputhit11.cov = Matrix<float,6,1>(tmphc11);

AHIT inputhit12;
const float tmphp12[3] = {9.38765717, 35.4993019, -109.677956};
const float tmphc12[6] = {0.797379732, 2.12806249, 5.67952585, -0, 0, 0};
inputhit12.pos = Matrix<float,3,1>(tmphp12);
inputhit12.cov = Matrix<float,6,1>(tmphc12);

AHIT inputhit13;
const float tmphp13[3] = {11.0634117, 43.4505577, -134.310318};
const float tmphc13[6] = {0.623921394, 2.45037341, 9.62381172, -0, 0, 0};
inputhit13.pos = Matrix<float,3,1>(tmphp13);
inputhit13.cov = Matrix<float,6,1>(tmphc13);

AHIT inputhit14;
const float tmphp14[3] = {10.9692974, 43.4805908, -148.308456};
const float tmphc14[6] = {0.613398015, 2.43132401, 9.63731003, 0, -0, 0};
inputhit14.pos = Matrix<float,3,1>(tmphp14);
inputhit14.cov = Matrix<float,6,1>(tmphc14);

AHIT inputhit15;
const float tmphp15[3] = {13.6062117, 54.542469, -165.336273};
const float tmphc15[6] = {0.648314595, 2.59879255, 10.4175854, 3.26234263e-16, 1.30775005e-15, 1.64165685e-31};
inputhit15.pos = Matrix<float,3,1>(tmphp15);
inputhit15.cov = Matrix<float,6,1>(tmphc15);

AHIT inputhit16;
const float tmphp16[3] = {13.4874973, 54.576767, -179.334564};
const float tmphc16[6] = {0.636996567, 2.5776298, 10.4307089, 3.234741e-16, 1.30897825e-15, 1.64267283e-31};
inputhit16.pos = Matrix<float,3,1>(tmphp16);
inputhit16.cov = Matrix<float,6,1>(tmphc16);

AHIT inputhit17;
const float tmphp17[3] = {16.124157, 65.7403793, -189.884399};
const float tmphc17[6] = {1.02542591, 4.18091106, 17.0468922, 5.23224904e-16, 2.13335179e-15, 2.66980604e-31};
inputhit17.pos = Matrix<float,3,1>(tmphp17);
inputhit17.cov = Matrix<float,6,1>(tmphc17);

AHIT inputhit18;
const float tmphp18[3] = {15.4582443, 65.860466, -189.571487};
const float tmphc18[6] = {0.304583073, 2.32625961, 17.767849, -0, 0, 0};
inputhit18.pos = Matrix<float,3,1>(tmphp18);
inputhit18.cov = Matrix<float,6,1>(tmphc18);

AHIT inputhit19;
const float tmphp19[3] = {15.9467182, 65.7825699, -207.383026};
const float tmphc19[6] = {1.00305867, 4.13773108, 17.0689125, 0.00142698595, 0.00588658359, 2.03011541e-06};
inputhit19.pos = Matrix<float,3,1>(tmphp19);
inputhit19.cov = Matrix<float,6,1>(tmphc19);

AHIT inputhit20;
const float tmphp20[3] = {15.9135628, 65.7994232, -207.070282};
const float tmphc20[6] = {0.336847425, 2.44411731, 17.7350674, 0.000559302513, 0.00405845372, 9.28728525e-07};
inputhit20.pos = Matrix<float,3,1>(tmphp20);
inputhit20.cov = Matrix<float,6,1>(tmphc20);

AHIT inputhit21;
const float tmphp21[3] = {15.7538891, 65.829277, -226.381378};
const float tmphc21[6] = {0.978875101, 4.09043169, 17.0930214, 0.000899855106, 0.00376031385, 8.27236761e-07};
inputhit21.pos = Matrix<float,3,1>(tmphp21);
inputhit21.cov = Matrix<float,6,1>(tmphc21);

AHIT inputhit22;
const float tmphp22[3] = {19.0423851, 79.7255478, -229.755692};
const float tmphc22[6] = {1.52722645, 6.39435101, 26.7730618, -0.00174889574, -0.00732256798, 2.00276168e-06};
inputhit22.pos = Matrix<float,3,1>(tmphp22);
inputhit22.cov = Matrix<float,6,1>(tmphc22);

AHIT inputhit23;
const float tmphp23[3] = {16.2950249, 65.7468643, -226.068665};
const float tmphc23[6] = {0.365449488, 2.54378247, 17.7073212, -0.000267859956, -0.00186453806, 1.96333346e-07};
inputhit23.pos = Matrix<float,3,1>(tmphp23);
inputhit23.cov = Matrix<float,6,1>(tmphc23);

AHIT inputhit24;
const float tmphp24[3] = {18.7913284, 79.7989655, -250.253082};
const float tmphc24[6] = {1.48767769, 6.31668472, 26.821207, 0.00143374864, 0.00608785078, 1.38181554e-06};
inputhit24.pos = Matrix<float,3,1>(tmphp24);
inputhit24.cov = Matrix<float,6,1>(tmphc24);

AHIT inputhit25;
const float tmphp25[3] = {18.5258789, 79.8741455, -271.755951};
const float tmphc25[6] = {1.44613361, 6.23380566, 26.8724403, -0.00102105062, -0.00440146076, 7.20922571e-07};
inputhit25.pos = Matrix<float,3,1>(tmphp25);
inputhit25.cov = Matrix<float,6,1>(tmphc25);
