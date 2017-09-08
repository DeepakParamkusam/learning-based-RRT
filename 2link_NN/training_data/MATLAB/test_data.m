clear all;

data = load('testdata.mat');
testdata=data.testdata;

test_scaled = (testdata-min(testdata))./(max(testdata)-min(testdata));
test_std = (testdata-mean(testdata))./std(testdata);

test_scaled_std = (test_scaled-mean(test_scaled))./std(test_scaled);
test_std_scaled = (test_std-min(test_std))./(max(test_std)-min(test_std));

figure(1)
for i=1:3000
    k=waitforbuttonpress;
    subplot(3,2,1)
    plot(testdata(i,:))
    title('Actual')
    
    subplot(3,2,3)
    plot(test_scaled(i,:))
    title('Scaled')
    
    subplot(3,2,4)
    plot(test_std(i,:))
    title('Standardized')
    
    subplot(3,2,5)
    plot(test_scaled_std(i,:))
    title('Scaled -> Standardized')
    
    subplot(3,2,6)
    plot(test_std_scaled(i,:))
    title('Standardized -> Scaled')    
end

