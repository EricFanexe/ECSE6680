def fp2fx():
    with open('FIR_filter/b.txt','r') as f:
        b = f.read().split('\n')
        b_fx = []
        for i in range(len(b)-1):
            print(str(b[i])+',')
            b_fx.append(round(float(b[i])*32768))
        b_out = []
        for b_i in b_fx:
            if b_i>=0:
                #print("16\'b"+format(b_i,'b').zfill(16)+',')
                print(str(b_i)+',')
            elif b_i <0:
                #print("16\'b"+format(pow(2,16)-abs(b_i),'b').zfill(16)+',')
                print(str(b_i)+',')
        print(b_out)

    return 

if __name__ == "__main__":
    b = fp2fx()