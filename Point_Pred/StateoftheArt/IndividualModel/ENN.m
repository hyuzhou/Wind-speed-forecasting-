[pn,minp,maxp,tn,mint,maxt]=premnmx(TrainX',TrainY');
p2= tramnmx(TestX',minp,maxp);
net_1 = newelm(minmax(pn),[10,1],{'tansig','purelin'},'traingdm');
net.trainparam.show=100;
net.trainparam.epochs=3000;
net.trainparam.goal=0.0001;
net=init(net_1);
net = train(net,pn,tn);  
PN = sim(net,p2); 
TestResult= postmnmx(PN,mint,maxt);
E = TestOutput - TestResult 
MSE=mse(E);
figure(1)
plot(TestOutput,'bo-');
hold on;
plot(TestResult,'r*--');
legend('真实值','预测值');
save('Elman.mat','net');