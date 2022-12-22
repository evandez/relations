# Scan Results
Checked relations 
```
P17     => '{} is located in the country of',
P641    => '{} plays the sport of',
P103    => 'Mother language of {} is',
P176    => '{} is produced by'
```
For each relation
* First, find all the *"good"* cases in counterfact that the model predicts the object correctly with the same relation prompt
* Sample a number of `test cases` from those good cases
* Calculate Jacobian and bias for each of the good cases and evaluate it against the `test cases`
    * `h_idx` at the last subject token index
        * didn't find any good cases
    * `h_idx` at **ALL** of the subject token indices
        * found some good cases
    * Calculate `J` and `bias` after final layer norm `ln_f`
        * Not getting really good results. <br>
        `The Space Needle` case
        ```
        Niagara Falls, target: Canada   ==>   predicted: [' Niagara', 'Toronto', ' Ontario', ' Cuomo', ' Erie']
        Valdemarsvik, target: Sweden   ==>   predicted: [' Nordic', ' Greenland', 'vik', ' Swedish', ' Icelandic']
        Kyoto University, target: Japan   ==>   predicted: ['Japanese', ' Japanese', 'Tok', 'Japan', ' Japan']
        Hattfjelldal, target: Norway   ==>   predicted: [' Nordic', ' Denmark', ' Iceland', ' Scandinavian', ' Icelandic']
        Ginza, target: Japan   ==>   predicted: [' Tokyo', 'Tok', ' Japan', 'Japan', ' Japanese']
        Sydney Hospital, target: Australia   ==>   predicted: [' Sydney', ' NSW', ' Australia', ' Australian', 'Australia']
        ```
    * consider $ z_{est} = h + Ah + bias $, where $A$ is a low rank matrix
        * Not getting very good results. Tried upto $rank = \{1,2,3\}$ approximations.

## Find out some combination that kind of works?
* Multiply Jacobian contribution $Jh$ by 3 or 4 --> Not working
* Average out all the calculated weights and biases --> Not working
* Average out the **good** cases --> Kind of working

## Which contributes the most? `Jh` or `bias`?
**Bias**<br>
Relation_id: `P17`
```
`Jh` contribution = 17.883 +/- 10.813
`bias` contribution = 449.510 +/- 50.288
```

```
`h + Ah` contribution = 176.677 +/- 211.293
`bias` contribution = 451.781 +/- 204.084
```


## What do the bias vectors represent in the embedding space?
**The original target**. <br>
Some examples (calculated at the last subject token, relation id: `P17`)

```
(Autonomous University of Madrid, {}, which is located in, Spain)
[' Spain', ' And', ' Catalonia', ' Gran', ' Se']
(Kuala Langat, {}, located in, Malaysia)
[' Malaysia', ' Indonesia', ' B', ' Thailand', ' Brune']
(Bastille, {}, which is located in, France)
[' France', ' the', ' Brittany', ' Bast', ' Catalonia']
(Valdemarsvik, {}, which is located in, Sweden)
[' Sweden', ' Norway', ' Denmark', ' Iceland', ' Finland']
(Piper Verlag, {}, which is located in, Germany)
[' Germany', ' Austria', ' the', ' Luxembourg', ' Switzerland']
(Tehri Garhwal district, {}, in, India)
[' India', ' Nepal', ' Bh', ' Pakistan', ' Afghanistan']
```

## Difference between $z$ from usual computation and $z_{est}$ calculated with Jacobian and Bias?
```
The Great Wall, target: China
z_ =  [' China', ' the', ' Xin', ' Yun', ' J']
z_est =  [' China', ' Hong', ' Beijing', ' Chinese', ' Shen']
Distance =>  257.2456970214844

Niagara Falls, target: Canada
z_ =  [' Ontario', ' Canada', ' New', ' Quebec', ' Newfoundland']
z_est =  [' Canada', ' Ontario', ' Niagara', ' New', ' British']
Distance =>  245.4635467529297

Valdemarsvik, target: Sweden
z_ =  [' Sweden', ' Norway', ' Denmark', ' Iceland', ' Finland']
z_est =  [' Iceland', ' Denmark', ' Sweden', ' Finland', ' Norway']
Distance =>  251.4376220703125

Kyoto University, target: Japan
z_ =  [' Japan', ' Kyoto', ' the', ' South', ' Okinawa']
z_est =  [' Japan', ' Japanese', ' Finland', ' Hawaii', ' Tokyo']
Distance =>  236.29151916503906

Hattfjelldal, target: Norway
z_ =  [' Norway', ' Iceland', ' Denmark', ' Sweden', ' Finland']
z_est =  [' Iceland', ' Denmark', ' Norway', ' Sweden', ' Finland']
Distance =>  270.375732421875

Ginza, target: Japan
z_ =  [' Japan', ' Tokyo', ' Gin', ' the', ' Sh']
z_est =  [' Japan', ' Singapore', ' China', ' Seattle', ' Hong']
Distance =>  241.63589477539062

Sydney Hospital, target: Australia
z_ =  [' Australia', ' New', ' the', ' Queensland', ' Papua']
z_est =  [' Australia', ' Sydney', ' Australian', ' Singapore', ' Canberra']
Distance =>  237.59640502929688

Mahalangur Himal, target: Nepal
z_ =  [' Nepal', ' Bh', ' India', ' Tibet', ' Utt']
z_est =  [' Nepal', ' Tibet', ' Bh', ' Nep', ' China']
Distance =>  330.04510498046875

Higashikagawa, target: Japan
z_ =  [' Japan', ' Fuk', ' Okinawa', ' Sh', ' Hok']
z_est =  [' Japan', ' Japanese', ' Tokyo', ' Canada', ' Seattle']
Distance =>  274.65643310546875

Trento, target: Italy
z_ =  [' Italy', ' Malta', ' T', ' Spain', ' the']
z_est =  [' Sweden', ' Finland', ' Iceland', ' Denmark', ' Luxembourg']
Distance =>  261.7779846191406

Taj Mahal, target: India
z_ =  [' India', ' Oman', ' Nepal', ' Taj', ' Pakistan']
z_est =  [' Iceland', ' Canada', ' Finland', ' Norway', ' New']
Distance =>  260.48834228515625
```

## How different `J` and `bias` values are for the good cases?

## Weight

### good cases
```
12.594758987426758  [0.0, 16.952, 18.875, 27.862, 17.031, 19.042]
17.02851104736328   [16.952, 0.0, 19.567, 28.279, 18.751, 20.4]
18.64788246154785   [18.875, 19.567, 0.0, 27.658, 20.576, 20.895]
28.792652130126953  [27.862, 28.279, 27.658, 0.0, 28.578, 24.671]
15.542061805725098  [17.031, 18.751, 20.576, 28.578, 0.0, 19.969]
18.99873924255371   [19.042, 20.4, 20.895, 24.671, 19.969, 0.0]
```
### Bad cases
```
4.320530891418457   [0.0, 6.46, 10.998, 9.945, 6.528, 13.408]
5.322466850280762   [6.46, 0.0, 11.687, 10.626, 7.393, 14.152]
11.044798851013184  [10.998, 11.687, 0.0, 12.696, 11.647, 15.394]
9.669387817382812   [9.945, 10.626, 12.696, 0.0, 10.527, 14.935]
5.642831325531006   [6.528, 7.393, 11.647, 10.527, 0.0, 13.834]
13.551694869995117  [13.408, 14.152, 15.394, 14.935, 13.834, 0.0]
```

## Bias
### good cases
```
486.58489990234375  [0.0, 128.522, 188.032, 205.003, 194.208, 189.796]
522.3410034179688   [128.522, 0.0, 214.299, 230.062, 187.036, 215.347]
430.8843994140625   [188.032, 214.299, 0.0, 197.953, 236.528, 191.542]
443.0304870605469   [205.003, 230.062, 197.953, 0.0, 245.654, 167.442]
517.58056640625     [194.208, 187.036, 236.528, 245.654, 0.0, 211.585]
446.3467102050781   [189.796, 215.347, 191.542, 167.442, 211.585, 0.0]
```

### bad cases
```
448.4342956542969   [0.0, 252.056, 242.965, 205.957, 165.566, 244.018]
484.4488830566406   [252.056, 0.0, 235.024, 206.688, 250.256, 248.271]
523.7765502929688   [242.965, 235.024, 0.0, 230.195, 204.277, 265.462]
392.2934265136719   [205.957, 206.688, 230.195, 0.0, 202.329, 224.214]
467.7102966308594   [165.566, 250.256, 204.277, 202.329, 0.0, 227.118]
449.6788330078125   [244.018, 248.271, 265.462, 224.214, 227.118, 0.0]
```


