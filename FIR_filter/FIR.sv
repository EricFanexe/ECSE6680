module FIR (
    input logic clk,
    input logic reset,
    input logic signed [32:0] x,
    output logic signed [32:0] y
);

// Coefficients for the FIR filter
parameter [32:0] coeff[128:0] = 
    '{4,13,30,57,91,126,146,134,70,-62,-266,-527,-811,-1069,-1244,-1288,-1174,-908,-536,-131,219,438,482,353,104,-179,-400,
    -487,-409,-195,81,320,438,390,192,-83,-333,-460,-410,-195,108,384,520,452,193,-168,-493,-642,-537,-192,281,701,883,711,192,
    -523,-1176,-1472,-1174,-191,1376,3245,5017,6284,6744,6284,5017,3245,1376,-191,-1174,-1472,-1176,-523,192,711,883,701,281,
    -192,-537,-642,-493,-168,193,452,520,384,108,-195,-410,-460,-333,-83,192,390,438,320,81,-195,-409,-487,-400,-179,104,353,
    482,438,219,-131,-536,-908,-1174,-1288,-1244,-1069,-811,-527,-266,-62,70,134,146,126,91,57,30,13,4};
// Internal pipeline registers
logic signed [32:0] delay [127:0];

logic signed [32:0] y_reg;
integer i,j;

always_ff @(posedge clk or posedge reset) begin
    if (reset) begin

        delay <= '{default: 32'd0};

    end else begin
        delay[0] <= x;
        for (i=1;i<128;i=i+1)begin
            delay[i] <= delay[i-1] ;
        end

    end
end
always_comb begin

y_reg = delay[127] * coeff[0] + delay[126] * coeff[1] + delay[125] * coeff[2] + delay[124] * coeff[3]
+ delay[123] * coeff[4] + delay[122] * coeff[5] + delay[121] * coeff[6] + delay[120] * coeff[7]
+ delay[119] * coeff[8] + delay[118] * coeff[9] + delay[117] * coeff[10] + delay[116] * coeff[11]
+ delay[115] * coeff[12] + delay[114] * coeff[13] + delay[113] * coeff[14] + delay[112] * coeff[15]
+ delay[111] * coeff[16] + delay[110] * coeff[17] + delay[109] * coeff[18] + delay[108] * coeff[19]
+ delay[107] * coeff[20] + delay[106] * coeff[21] + delay[105] * coeff[22] + delay[104] * coeff[23]
+ delay[103] * coeff[24] + delay[102] * coeff[25] + delay[101] * coeff[26] + delay[100] * coeff[27]
+ delay[99] * coeff[28] + delay[98] * coeff[29] + delay[97] * coeff[30] + delay[96] * coeff[31]
+ delay[95] * coeff[32] + delay[94] * coeff[33] + delay[93] * coeff[34] + delay[92] * coeff[35]
+ delay[91] * coeff[36] + delay[90] * coeff[37] + delay[89] * coeff[38] + delay[88] * coeff[39]
+ delay[87] * coeff[40] + delay[86] * coeff[41] + delay[85] * coeff[42] + delay[84] * coeff[43]
+ delay[83] * coeff[44] + delay[82] * coeff[45] + delay[81] * coeff[46] + delay[80] * coeff[47]
+ delay[79] * coeff[48] + delay[78] * coeff[49] + delay[77] * coeff[50] + delay[76] * coeff[51]
+ delay[75] * coeff[52] + delay[74] * coeff[53] + delay[73] * coeff[54] + delay[72] * coeff[55]
+ delay[71] * coeff[56] + delay[70] * coeff[57] + delay[69] * coeff[58] + delay[68] * coeff[59]
+ delay[67] * coeff[60] + delay[66] * coeff[61] + delay[65] * coeff[62] + delay[64] * coeff[63]
+ delay[63] * coeff[64] + delay[62] * coeff[65] + delay[61] * coeff[66] + delay[60] * coeff[67]
+ delay[59] * coeff[68] + delay[58] * coeff[69] + delay[57] * coeff[70] + delay[56] * coeff[71]
+ delay[55] * coeff[72] + delay[54] * coeff[73] + delay[53] * coeff[74] + delay[52] * coeff[75]
+ delay[51] * coeff[76] + delay[50] * coeff[77] + delay[49] * coeff[78] + delay[48] * coeff[79]
+ delay[47] * coeff[80] + delay[46] * coeff[81] + delay[45] * coeff[82] + delay[44] * coeff[83]
+ delay[43] * coeff[84] + delay[42] * coeff[85] + delay[41] * coeff[86] + delay[40] * coeff[87]
+ delay[39] * coeff[88] + delay[38] * coeff[89] + delay[37] * coeff[90] + delay[36] * coeff[91]
+ delay[35] * coeff[92] + delay[34] * coeff[93] + delay[33] * coeff[94] + delay[32] * coeff[95]
+ delay[31] * coeff[96] + delay[30] * coeff[97] + delay[29] * coeff[98] + delay[28] * coeff[99]
+ delay[27] * coeff[100] + delay[26] * coeff[101] + delay[25] * coeff[102] + delay[24] * coeff[103]
+ delay[23] * coeff[104] + delay[22] * coeff[105] + delay[21] * coeff[106] + delay[20] * coeff[107]
+ delay[19] * coeff[108] + delay[18] * coeff[109] + delay[17] * coeff[110] + delay[16] * coeff[111]
+ delay[15] * coeff[112] + delay[14] * coeff[113] + delay[13] * coeff[114] + delay[12] * coeff[115]
+ delay[11] * coeff[116] + delay[10] * coeff[117] + delay[9] * coeff[118] + delay[8] * coeff[119]
+ delay[7] * coeff[120] + delay[6] * coeff[121] + delay[5] * coeff[122] + delay[4] * coeff[123]
+ delay[3] * coeff[124] + delay[2] * coeff[125] + delay[1] * coeff[126] + delay[0] * coeff[127]+ x*coeff[128];

end


assign y = y_reg;

endmodule
