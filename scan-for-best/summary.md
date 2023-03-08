# Scan Results
Checked relations 
```
P17     => '{} is located in the country of',
P641    => '{} plays the sport of',
P103    => 'Mother language of {} is',
P176    => '{} is produced by'
```
For each relation
* First, find all the cases where model (`gpt-j`) correctly predicts the target object ==> `filtered cases`.
* Evaluation Strategy: Sample 50 `test cases` randomly from the `filtered cases` and use them for evaluating how good the calculated $J$ and $bias$ are (checking precision @ top 5 predictions).
* Calculate Jacobian and bias for each of the filtered cases and evaluate it against the `test cases`. $J$ and $bias$ are calculated at the $h_{idx}$ token position and after some middle layer of the model. ($15^{th}$ layer for `gpt-j`) 
    * $h_{idx}$ at the last subject token index
        * didn't find any good cases 
    * $h_{idx}$ at **ALL** of the subject token indices
        * found some good cases (some good cases for relation `P17` are at the bottom of the doc)
    * Calculate $J$ and $bias$ after final layer norm `ln_f`
        * Predicted object seems a little distorted. <br>
        If $J$ and $bias$ were calculated wiht the `The Space Needle` case
        ```
        Niagara Falls, target: Canada   ==>   predicted: [' Niagara', 'Toronto', ' Ontario', ' Cuomo', ' Erie']
        Valdemarsvik, target: Sweden   ==>   predicted: [' Nordic', ' Greenland', 'vik', ' Swedish', ' Icelandic']
        Kyoto University, target: Japan   ==>   predicted: ['Japanese', ' Japanese', 'Tok', 'Japan', ' Japan']
        Hattfjelldal, target: Norway   ==>   predicted: [' Nordic', ' Denmark', ' Iceland', ' Scandinavian', ' Icelandic']
        Ginza, target: Japan   ==>   predicted: [' Tokyo', 'Tok', ' Japan', 'Japan', ' Japanese']
        Sydney Hospital, target: Australia   ==>   predicted: [' Sydney', ' NSW', ' Australia', ' Australian', 'Australia']
        ```
    * consider $ z_{est} = h + Ah + bias $, where $A$ is a low rank matrix
        * Tried with `gpt2-xl`. Did not get very good results. Tried upto $rank = \{1,2,3\}$ approximations.

## Find out some combination that kind of works?
* Multiply Jacobian contribution $Jh$ by 3 or 4 --> Not working
* Average out all the calculated weights and biases --> Not working
* Average out the **good** cases --> Kind of working (check Evan's slides)

## Which contributes the most? $Jh$ or $bias$? (L2 norms)
**Bias**<br>
Relation_id: `P17`
```
`Jh` contribution = 12.675 +/- 8.163
`bias` contribution = 254.114 +/- 23.259
```
## Rank of jacobian weights
`3788.0519 +/- 115.542`
Inner representation dimensions for `gpt-j` is **4096**

## What do the bias vectors represent in the embedding space?
**The original target**. <br>
Some examples (calculated at the last subject token, relation id: `P17`)

```
(Autonomous University of Madrid, {}, which is located in, Spain)
[' Spain', ' Madrid', ' the', ' And', ' Cast']
(Kuala Langat, {}, located in, Malaysia)
[' Malaysia', ' Pen', ' Sel', ' Sab', ' Mal']
(Wanne-Eickel Central Station, {}, located in, Germany)
[' Germany', ' Han', ' North', ' Bad', ' Sch']
(Bastille, {}, which is located in, France)
[' France', ' the', ' Belgium', ' Luxembourg', ' Brittany']
(Shablykinsky District, {} is located in the country of, Russia)
[' Russia', ' Ukraine', ' Kazakhstan', ' Belarus', ' the']
(Valdemarsvik, {}, which is located in, Sweden)
[' Sweden', ' Latvia', ' S', ' J', ' Estonia']
(Attingal, {}, which is located in, India)
[' India', ' Kerala', ' Go', ' And', ' Karn']
(Nizampatnam, {} is located in the country of, India)
[' And', ' India', ' Od', ' Tel', ' Tamil']
```

## Difference between $z$ from usual computation and $z_{est}$ calculated with Jacobian and Bias?
$J$ and $bias$ were calculated using the `The Space Needle is located in the country of` case.
```
The Great Wall, target: China
z_ =  [' China', ' G', ' Shan', ' the', ' Great']
z_est =  [' China', ' Seattle', ' Beijing', ' Taiwan', ' Japan']
Distance =>  169.125

Niagara Falls, target: Canada
z_ =  [' Canada', ' Ontario', ' New', ' the', ' Niagara']
z_est =  [' Niagara', ' Canada', 'Ni', ' New', ' Ontario']
Distance =>  180.25

Valdemarsvik, target: Sweden
z_ =  [' Sweden', ' S', ' J', ' Latvia', ' V']
z_est =  [' Sweden', ' Seattle', ' Washington', ' Scandinav', ' Swedish']
Distance =>  179.25

Kyoto University, target: Japan
z_ =  [' Japan', ' Kyoto', ' Ky', ' the', ' H']
z_est =  [' Kyoto', ' Japan', ' Osaka', ' Tokyo', ' Japanese']
Distance =>  163.25

Hattfjelldal, target: Norway
z_ =  [' Norway', ' Nord', ' S', ' Tr', ' Finn']
z_est =  [' Norway', ' Sweden', ' Scandinav', ' Denmark', ' Washington']
Distance =>  189.75

Ginza, target: Japan
z_ =  [' Japan', ' Tokyo', ' the', ' Ch', ' Shin']
z_est =  [' Japan', ' Tokyo', ' Seattle', ' Osaka', ' Shin']
Distance =>  150.75

Sydney Hospital, target: Australia
z_ =  [' Australia', ' New', ' the', ' Sydney', ' NSW']
z_est =  [' Australia', ' Sydney', ' Australian', ' Queensland', ' New']
Distance =>  161.5

Mahalangur Himal, target: Nepal
z_ =  [' Nepal', ' India', ' Bh', ' Sik', ' Utt']
z_est =  [' Nepal', ' Seattle', ' Washington', ' Bh', ' Switzerland']
Distance =>  200.375

Higashikagawa, target: Japan
z_ =  [' Japan', ' Nag', ' Wak', ' Sh', ' the']
z_est =  [' Japan', ' Tokyo', ' Seattle', ' Osaka', ' Kyoto']
Distance =>  166.0

Trento, target: Italy
z_ =  [' Trent', ' Italy', ' the', ' Fri', ' Ven']
z_est =  [' Italy', ' Seattle', ' Washington', ' Sweden', ' Switzerland']
Distance =>  186.625

Taj Mahal, target: India
z_ =  [' India', ' Pakistan', ' Uttar', ' the', ' Bangladesh']
z_est =  [' Seattle', ' Washington', ' the', ' Sultan', ' Wa']
Distance =>  172.5
```

## How different $J$ and $bias$ values are for the good cases?
Not seeing any numerical patterns to distinguish the good cases from the bad cases. <br/>
**norm [pairwise L2 distances]**
## Weight

### good cases
```
19.188  [0.0,    38.375, 21.688, 18.078, 23.047, 21.859, 26.297]
39.75   [38.375, 0.0,    36.188, 38.625, 38.406, 40.906, 38.406]
20.594  [21.688, 36.188, 0.0,    21.422, 23.797, 24.375, 26.688]
15.312  [18.078, 38.625, 21.422, 0.0,    22.859, 19.344, 24.812]
23.0    [23.047, 38.406, 23.797, 22.859, 0.0,    25.266, 27.953]
18.828  [21.859, 40.906, 24.375, 19.344, 25.266, 0.0,    27.016]
26.5    [26.297, 38.406, 26.688, 24.812, 27.953, 27.016, 0.0   ]
```
### Bad cases
```
17.549  [0.0,    17.949, 19.386, 27.599, 17.965, 16.597, 23.482]
9.679   [17.949, 0.0,    17.015, 27.033, 10.177, 10.039, 19.596]
16.475  [19.386, 17.015, 0.0,    28.162, 17.053, 15.313, 22.612]
27.394  [27.599, 27.033, 28.162, 0.0,    27.206, 26.912, 31.171]
8.184   [17.965, 10.177, 17.053, 27.206, 0.0,     8.961, 18.904]
5.596   [16.597, 10.039, 15.313, 26.912, 8.961,  0.0,    18.207]
18.04   [23.482, 19.596, 22.612, 31.171, 18.904, 18.207, 0.0   ]
```

## Bias
### good cases
```
269.75  [0.0,     178.0,   92.938,  92.312, 107.188, 131.125, 137.125]
230.5   [178.0,   0.0,     174.125, 173.5,  186.375, 188.25,  172.75]
253.25  [92.938,  174.125, 0.0,     102.312, 115.125, 131.375, 139.5]
247.25  [92.312,  173.5,   102.312, 0.0,    107.188, 124.312, 122.062]
274.25  [107.188, 186.375, 115.125, 107.188, 0.0,    134.625, 150.25]
246.0   [131.125, 188.25,  131.375, 124.312, 134.625, 0.0,  137.125]
216.75  [137.125, 172.75,  139.5,   122.062, 150.25, 137.125, 0.0]
```

### bad cases
```
280.816 [0.0, 161.95, 138.882, 134.235, 178.476, 129.226, 172.651]
223.301 [161.95, 0.0, 163.701, 160.271, 165.099, 167.165, 176.24]
259.589 [138.882, 163.701, 0.0, 160.631, 180.087, 109.962, 180.799]
273.735 [134.235, 160.271, 160.631, 0.0, 173.806, 157.031, 169.659]
237.356 [178.476, 165.099, 180.087, 173.806, 0.0, 181.295, 173.081]
250.456 [129.226, 167.165, 109.962, 157.031, 181.295, 0.0, 174.918]
240.872 [172.651, 176.24, 180.799, 169.659, 173.081, 174.918, 0.0]
```


---
# Some good cases (`P17`)
```
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
(s = Haut Atlas, r = {} is located in the country of [P17], o = Morocco)
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
----------------------------------------------------------------------------------------------------
{'Jh_norm': 12.203125, 'bias_norm': 269.75, 'h_info': {'h_index': 2, 'token_id': 22494, 'token': ' Atlas'}, 'consider_residual': False}
----------------------------------------------------------------------------------------------------
Emscher, target: Germany   ==>   predicted: [' Germany', ' Lie', ' Luxembourg', ' Switzerland', ' Morocco'] >>> True
Umarex, target: Germany   ==>   predicted: [' Switzerland', ' Lie', ' France', ' Germany', ' Morocco'] >>> True
Gavrilovo-Posadsky District, target: Russia   ==>   predicted: [' Georgia', ' Kazakhstan', ' Belarus', ' Ukraine', ' the'] >>> False
Cairo American College, target: Egypt   ==>   predicted: [' Morocco', ' the', ' Switzerland', ' Lebanon', ' Qatar'] >>> False
Fort Madalena, target: Malta   ==>   predicted: [' Morocco', ' Chile', ' And', ' Maurit', ' Spain'] >>> False
College of Engineering, Pune, target: India   ==>   predicted: [' India', ' Bh', ' Morocco', ' Pakistan', ' Maurit'] >>> True
Staatliche Antikensammlungen, target: Germany   ==>   predicted: [' Lie', ' Germany', ' Morocco', ' Switzerland', ' Luxembourg'] >>> True
War in Donbass, target: Ukraine   ==>   predicted: [' Georgia', ' Belarus', ' Ukraine', ' Kazakhstan', ' Russia'] >>> True
Pannonhalma Archabbey, target: Hungary   ==>   predicted: [' Lie', ' Luxembourg', ' And', ' Austria', ' Switzerland'] >>> False
Roman Catholic Archdiocese of Lucca, target: Italy   ==>   predicted: [' Italy', ' And', ' Morocco', ' Switzerland', ' France'] >>> True
Harrington Sound, Bermuda, target: Bermuda   ==>   predicted: [' Morocco', ' Maurit', ' Atlas', ' Bermuda', ' Gibraltar'] >>> True
Mukkam, target: India   ==>   predicted: [' Morocco', ' Maurit', ' Oman', ' the', ' Lebanon'] >>> False
Pesisir Selatan, target: Indonesia   ==>   predicted: [' Morocco', ' Oman', ' Atlas', ' Maurit', ' Qatar'] >>> False
County Carlow, target: Ireland   ==>   predicted: [' Ireland', ' France', ' Georgia', ' Morocco', ' And'] >>> True
Iximche, target: Guatemala   ==>   predicted: [' Morocco', ' Algeria', ' Chad', ' Bolivia', ' Mali'] >>> False
Circuito da Boavista, target: Portugal   ==>   predicted: [' Morocco', ' And', ' Maurit', ' Atlas', ' France'] >>> False
Chu Lai Base Area, target: Vietnam   ==>   predicted: [' Nam', ' the', ' Lie', ' Switzerland', ' Morocco'] >>> False
Bugis Junction, target: Singapore   ==>   predicted: [' Morocco', ' Oman', ' Switzerland', ' Qatar', ' Australia'] >>> False
Giurgiu County, target: Romania   ==>   predicted: [' Morocco', ' Georgia', ' Algeria', ' the', ' Lie'] >>> False
Taksim Military Barracks, target: Turkey   ==>   predicted: [' Turkey', ' Morocco', ' Lie', ' Georgia', ' the'] >>> True
Gmina Konarzyny, target: Poland   ==>   predicted: [' Morocco', ' the', ' Germany', ' Poland', ' Lie'] >>> True
Bahujan Vikas Aaghadi, target: India   ==>   predicted: [' Morocco', ' India', ' Maurit', ' the', ' Switzerland'] >>> True
Hyderabad Deccan railway station, target: India   ==>   predicted: [' Switzerland', ' India', ' France', ' Bh', ' the'] >>> True
Cappoquin, target: Ireland   ==>   predicted: [' Ireland', ' And', ' France', ' Luxembourg', ' Morocco'] >>> True
Annweiler am Trifels, target: Germany   ==>   predicted: [' Lie', ' Germany', ' Luxembourg', ' Switzerland', ' Morocco'] >>> True
Kuala Langat, target: Malaysia   ==>   predicted: [' Morocco', ' Atlas', ' Oman', ' Qatar', ' Maurit'] >>> False
plaza de Cibeles, target: Spain   ==>   predicted: [' Spain', ' Morocco', ' And', ' France', ' Belgium'] >>> True
Bilecik Province, target: Turkey   ==>   predicted: [' Morocco', ' Turkey', ' Algeria', ' Lebanon', ' Jordan'] >>> True
Mittag-Leffler Institute, target: Sweden   ==>   predicted: [' Lie', ' Switzerland', ' Luxembourg', ' the', ' Georgia'] >>> False
Shibganj Upazila, Bogra District, target: Bangladesh   ==>   predicted: [' Morocco', ' the', ' Maurit', ' Dj', ' Mali'] >>> False
San Canzian d'Isonzo, target: Italy   ==>   predicted: [' Lie', ' Morocco', ' Italy', ' And', ' Switzerland'] >>> True
Dwarka, target: India   ==>   predicted: [' India', ' Morocco', ' Oman', ' Bh', ' Maurit'] >>> True
Hultsfred Municipality, target: Sweden   ==>   predicted: [' Denmark', ' Lie', ' Sweden', ' Switzerland', ' Germany'] >>> True
Argentine National Anthem, target: Argentina   ==>   predicted: [' Lie', ' the', ' France', ' Belgium', ' Switzerland'] >>> False
Bolpur, target: India   ==>   predicted: [' India', ' Bh', ' Morocco', ' Afghanistan', ' Nepal'] >>> True
Battle of Montereau, target: France   ==>   predicted: [' France', ' Luxembourg', ' Belgium', ' Switzerland', ' Lie'] >>> True
Hattfjelldal, target: Norway   ==>   predicted: [' Norway', ' Morocco', ' Denmark', ' Sweden', ' Switzerland'] >>> True
Guillaumes, target: France   ==>   predicted: [' France', ' Belgium', ' Luxembourg', ' Morocco', ' And'] >>> True
Kowsar County, target: Iran   ==>   predicted: [' Morocco', ' Algeria', ' Jordan', ' Lebanon', ' Oman'] >>> False
Grobbendonk, target: Belgium   ==>   predicted: [' Belgium', ' Luxembourg', ' Lie', ' Morocco', ' Nam'] >>> True
Japan National Route 112, target: Japan   ==>   predicted: [' Japan', ' France', ' Switzerland', ' Lie', ' the'] >>> True
Lismore GAA, target: Ireland   ==>   predicted: [' Ireland', ' And', ' Luxembourg', ' France', ' Georgia'] >>> True
Band-e Kaisar, target: Iran   ==>   predicted: [' Morocco', ' Oman', ' Lebanon', ' Jordan', ' Algeria'] >>> False
Mbale District, target: Uganda   ==>   predicted: [' Burk', ' Rwanda', ' Ethiopia', ' Dj', ' Mali'] >>> False
Penna Ahobilam, target: India   ==>   predicted: [' Morocco', ' Oman', ' Maurit', ' India', ' the'] >>> True
Indira Gandhi International Airport, target: India   ==>   predicted: [' Morocco', ' Switzerland', ' Qatar', ' Oman', ' Maurit'] >>> False
Peremyshliany, target: Ukraine   ==>   predicted: [' Belarus', ' Georgia', ' Morocco', ' Ukraine', ' Kazakhstan'] >>> True
Central Black Forest, target: Germany   ==>   predicted: [' Lie', ' Germany', ' Switzerland', ' Luxembourg', ' Austria'] >>> True
Nishi-Matsuura District, target: Japan   ==>   predicted: [' Japan', ' the', ' Morocco', ' Nam', ' France'] >>> True
Canada Live, target: Canada   ==>   predicted: [' Canada', ' Quebec', ' Belgium', ' Morocco', ' Switzerland'] >>> True
----------------------------------------------------------------------------------------------------
31/50
----------------------------------------------------------------------------------------------------


zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
(s = Pamukkale, r = {} is located in the country of [P17], o = Turkey)
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
----------------------------------------------------------------------------------------------------
{'Jh_norm': 23.96875, 'bias_norm': 230.5, 'h_info': {'h_index': 4, 'token_id': 1000, 'token': 'ale'}, 'consider_residual': False}
----------------------------------------------------------------------------------------------------
Emscher, target: Germany   ==>   predicted: [' Germany', ' German', ' North', ' Northern', ' Den'] >>> True
Umarex, target: Germany   ==>   predicted: [' Den', ' Turkey', ' Thr', ' Germany', ' Sam'] >>> True
Gavrilovo-Posadsky District, target: Russia   ==>   predicted: [' Thr', ' Turkey', ' Den', ' Bulgaria', ' Georgia'] >>> False
Cairo American College, target: Egypt   ==>   predicted: [' Cyprus', ' Turkey', ' Den', ' Thr', ' western'] >>> False
Fort Madalena, target: Malta   ==>   predicted: [' Cyprus', ' Cre', ' Malta', ' Croatia', ' Turkey'] >>> True
College of Engineering, Pune, target: India   ==>   predicted: [' India', ' Maharashtra', ' Pakistan', ' Turkey', ' Punjab'] >>> True
Staatliche Antikensammlungen, target: Germany   ==>   predicted: [' Turkey', ' Den', ' M', ' Thr', ' Iz'] >>> False
War in Donbass, target: Ukraine   ==>   predicted: [' Ukraine', ' Thr', ' Crimea', ' Turkey', ' Northern'] >>> True
Pannonhalma Archabbey, target: Hungary   ==>   predicted: [' Turkey', ' B', ' C', ' Hungary', ' Cyprus'] >>> True
Roman Catholic Archdiocese of Lucca, target: Italy   ==>   predicted: [' Italy', ' Turkey', ' Kang', ' Sicily', ' South'] >>> True
Harrington Sound, Bermuda, target: Bermuda   ==>   predicted: [' Bermuda', ' Cyprus', ' Berm', ' Malta', ' Island'] >>> True
Mukkam, target: India   ==>   predicted: [' Turkey', ' M', ' South', ' Mu', ' K'] >>> False
Pesisir Selatan, target: Indonesia   ==>   predicted: [' Turkey', ' South', ' M', ' Den', ' Indonesia'] >>> True
County Carlow, target: Ireland   ==>   predicted: [' Ireland', 'Ireland', ' Georgia', ' Irish', ' Turkey'] >>> True
Iximche, target: Guatemala   ==>   predicted: [' Turkey', ' C', ' the', ' Den', ' western'] >>> False
Circuito da Boavista, target: Portugal   ==>   predicted: [' Turkey', ' Man', ' North', ' Northern', ' Cape'] >>> False
Chu Lai Base Area, target: Vietnam   ==>   predicted: [' Den', ' Cyprus', ' B', ' C', ' Man'] >>> False
Bugis Junction, target: Singapore   ==>   predicted: [' J', ' Java', ' Thr', ' Jordan', ' Turkey'] >>> False
Giurgiu County, target: Romania   ==>   predicted: [' Romania', ' Turkey', ' Thr', ' Georgia', ' Den'] >>> True
Taksim Military Barracks, target: Turkey   ==>   predicted: [' Thr', ' Mar', ' Turkey', ' Den', ' Mu'] >>> True
Gmina Konarzyny, target: Poland   ==>   predicted: [' Poland', ' Turkey', ' Den', ' Cyprus', ' C'] >>> True
Bahujan Vikas Aaghadi, target: India   ==>   predicted: [' Turkey', ' South', ' Den', ' C', ' North'] >>> False
Hyderabad Deccan railway station, target: India   ==>   predicted: [' Turkey', ' Thr', ' South', ' Iz', ' Sam'] >>> False
Cappoquin, target: Ireland   ==>   predicted: [' Turkey', ' Ireland', ' Iz', ' �', ' I'] >>> True
Annweiler am Trifels, target: Germany   ==>   predicted: [' Germany', ' Turkey', ' North', ' C', ' Den'] >>> True
Kuala Langat, target: Malaysia   ==>   predicted: [' Turkey', ' Man', ' South', ' M', ' Malaysia'] >>> True
plaza de Cibeles, target: Spain   ==>   predicted: [' Spain', ' Turkey', ' �', ' Iz', ' Europe'] >>> True
Bilecik Province, target: Turkey   ==>   predicted: [' Turkey', ' Thr', ' Den', ' Bulgaria', ' western'] >>> True
Mittag-Leffler Institute, target: Sweden   ==>   predicted: [' Iz', ' Turkey', ' Den', ' �', ' Cyprus'] >>> False
Shibganj Upazila, Bogra District, target: Bangladesh   ==>   predicted: [' Turkey', ' B', ' Thr', ' South', ' Den'] >>> True
San Canzian d'Isonzo, target: Italy   ==>   predicted: [' Turkey', ' Italy', ' Croatia', ' Thr', ' Macedonia'] >>> True
Dwarka, target: India   ==>   predicted: [' Turkey', ' India', ' Thr', ' Palestine', ' Pakistan'] >>> True
Hultsfred Municipality, target: Sweden   ==>   predicted: [' Thr', ' Turkey', ' Sweden', ' Den', ' South'] >>> True
Argentine National Anthem, target: Argentina   ==>   predicted: [' Thr', ' Turkey', ' Den', ' R', ' North'] >>> False
Bolpur, target: India   ==>   predicted: [' Bangladesh', ' Bengal', ' India', ' Turkey', ' B'] >>> True
Battle of Montereau, target: France   ==>   predicted: [' France', ' Turkey', ' Thr', ' Germany', ' Rou'] >>> True
Hattfjelldal, target: Norway   ==>   predicted: [' Denmark', ' North', ' Iceland', ' Norway', ' Dal'] >>> True
Guillaumes, target: France   ==>   predicted: [' Turkey', ' France', ' Thr', ' Greece', ' Les'] >>> True
Kowsar County, target: Iran   ==>   predicted: [' Turkey', ' C', ' Den', ' Georgia', ' San'] >>> False
Grobbendonk, target: Belgium   ==>   predicted: [' Netherlands', ' Belgium', ' Holland', ' Germany', ' Western'] >>> True
Japan National Route 112, target: Japan   ==>   predicted: [' Cyprus', ' Turkey', ' Den', ' Thr', ' Iz'] >>> False
Lismore GAA, target: Ireland   ==>   predicted: [' Ireland', ' Georgia', ' Irish', 'Ireland', ' G'] >>> True
Band-e Kaisar, target: Iran   ==>   predicted: [' Turkey', ' C', ' Anat', ' K', ' Iz'] >>> False
Mbale District, target: Uganda   ==>   predicted: [' Den', ' Thr', ' C', ' Turkey', ' South'] >>> False
Penna Ahobilam, target: India   ==>   predicted: [' Turkey', ' South', ' Thr', ' M', ' Iz'] >>> False
Indira Gandhi International Airport, target: India   ==>   predicted: [' Turkey', ' Thr', ' Den', ' Iz', ' K'] >>> False
Peremyshliany, target: Ukraine   ==>   predicted: [' Ukraine', ' Turkey', ' Bulgaria', ' Poland', ' Thr'] >>> True
Central Black Forest, target: Germany   ==>   predicted: [' Germany', ' Bad', ' German', ' Turkey', ' Switzerland'] >>> True
Nishi-Matsuura District, target: Japan   ==>   predicted: [' Den', ' Thr', ' Iz', ' Turkey', ' C'] >>> False
Canada Live, target: Canada   ==>   predicted: [' Canada', ' Manitoba', ' Canadian', ' C', ' Quebec'] >>> True
----------------------------------------------------------------------------------------------------
31/50
----------------------------------------------------------------------------------------------------


zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
(s = Fort Madalena, r = {} is located in the country of [P17], o = Malta)
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
----------------------------------------------------------------------------------------------------
{'Jh_norm': 13.3984375, 'bias_norm': 253.25, 'h_info': {'h_index': 3, 'token_id': 8107, 'token': 'ena'}, 'consider_residual': False}
----------------------------------------------------------------------------------------------------
Emscher, target: Germany   ==>   predicted: [' Germany', ' the', ' Italy', ' Luxembourg', ' San'] >>> True
Umarex, target: Germany   ==>   predicted: [' the', ' San', ' Chile', ' Portugal', ' Mexico'] >>> False
Gavrilovo-Posadsky District, target: Russia   ==>   predicted: [' the', ' Uruguay', ' San', ' Portugal', ' Chile'] >>> False
Cairo American College, target: Egypt   ==>   predicted: [' the', ' San', ' Mexico', ' Colombia', ' Georgia'] >>> False
Fort Madalena, target: Malta   ==>   predicted: [' Malta', ' the', ' San', ' Portugal', ' Italy'] >>> True
College of Engineering, Pune, target: India   ==>   predicted: [' India', ' Maurit', ' Maharashtra', ' Go', ' the'] >>> True
Staatliche Antikensammlungen, target: Germany   ==>   predicted: [' San', ' the', ' Portugal', ' Malta', ' Italy'] >>> False
War in Donbass, target: Ukraine   ==>   predicted: [' Malta', ' Ukraine', ' Monteneg', ' the', ' Lithuania'] >>> True
Pannonhalma Archabbey, target: Hungary   ==>   predicted: [' Malta', ' San', ' Monteneg', ' Croatia', ' And'] >>> False
Roman Catholic Archdiocese of Lucca, target: Italy   ==>   predicted: [' Italy', ' Sicily', ' Sard', ' San', ' T'] >>> True
Harrington Sound, Bermuda, target: Bermuda   ==>   predicted: [' Bermuda', ' Malta', ' St', ' Trinidad', ' Barb'] >>> True
Mukkam, target: India   ==>   predicted: [' Malta', ' the', ' Moz', ' Portugal', ' South'] >>> False
Pesisir Selatan, target: Indonesia   ==>   predicted: [' Indonesia', ' the', ' Malaysia', ' Malta', ' Van'] >>> True
County Carlow, target: Ireland   ==>   predicted: [' Ireland', ' Portugal', ' Malta', ' Dublin', ' Georgia'] >>> True
Iximche, target: Guatemala   ==>   predicted: [' Chile', ' Uruguay', ' San', ' Guatemala', ' Peru'] >>> True
Circuito da Boavista, target: Portugal   ==>   predicted: [' Portugal', ' Brazil', ' Angola', ' Rio', ' Cape'] >>> True
Chu Lai Base Area, target: Vietnam   ==>   predicted: [' the', ' San', ' Bel', ' B', ' Panama'] >>> False
Bugis Junction, target: Singapore   ==>   predicted: [' the', ' Bel', ' Guy', ' Australia', ' Panama'] >>> False
Giurgiu County, target: Romania   ==>   predicted: [' Romania', ' the', ' San', ' Portugal', ' Uruguay'] >>> True
Taksim Military Barracks, target: Turkey   ==>   predicted: [' the', ' Malta', ' And', ' San', ' Portugal'] >>> False
Gmina Konarzyny, target: Poland   ==>   predicted: [' Poland', ' Malta', ' Lithuania', ' the', ' San'] >>> True
Bahujan Vikas Aaghadi, target: India   ==>   predicted: [' the', ' Uruguay', ' Italy', ' San', ' Portugal'] >>> False
Hyderabad Deccan railway station, target: India   ==>   predicted: [' India', ' the', ' Portugal', ' Brazil', ' And'] >>> True
Cappoquin, target: Ireland   ==>   predicted: [' Portugal', ' Malta', ' Ireland', ' the', ' And'] >>> True
Annweiler am Trifels, target: Germany   ==>   predicted: [' San', ' Malta', ' the', ' Luxembourg', ' T'] >>> False
Kuala Langat, target: Malaysia   ==>   predicted: [' Malaysia', ' Malta', ' Singapore', ' Indonesia', ' Sri'] >>> True
plaza de Cibeles, target: Spain   ==>   predicted: [' Spain', ' And', ' Portugal', ' San', ' Chile'] >>> True
Bilecik Province, target: Turkey   ==>   predicted: [' Monteneg', ' the', ' Portugal', ' Uruguay', ' San'] >>> False
Mittag-Leffler Institute, target: Sweden   ==>   predicted: [' the', ' San', ' Chile', ' And', ' Portugal'] >>> False
Shibganj Upazila, Bogra District, target: Bangladesh   ==>   predicted: [' the', ' San', ' Uruguay', ' Bel', ' Colombia'] >>> False
San Canzian d'Isonzo, target: Italy   ==>   predicted: [' Italy', ' Malta', ' San', ' Sicily', ' Sard'] >>> True
Dwarka, target: India   ==>   predicted: [' India', ' Malta', ' the', ' Go', ' Dj'] >>> True
Hultsfred Municipality, target: Sweden   ==>   predicted: [' Sweden', ' the', ' Denmark', ' Chile', ' Portugal'] >>> True
Argentine National Anthem, target: Argentina   ==>   predicted: [' Georgia', ' the', ' Mexico', ' And', ' Portugal'] >>> False
Bolpur, target: India   ==>   predicted: [' India', ' Maurit', ' the', ' Moz', ' Sri'] >>> True
Battle of Montereau, target: France   ==>   predicted: [' France', ' Belgium', ' Luxembourg', ' Switzerland', ' Sweden'] >>> True
Hattfjelldal, target: Norway   ==>   predicted: [' Denmark', ' Norway', ' Sweden', ' Malta', ' the'] >>> True
Guillaumes, target: France   ==>   predicted: [' France', ' the', ' Belgium', ' Portugal', ' Chile'] >>> True
Kowsar County, target: Iran   ==>   predicted: [' the', ' Portugal', ' And', ' San', ' Chile'] >>> False
Grobbendonk, target: Belgium   ==>   predicted: [' Belgium', ' Portugal', ' Luxembourg', ' the', ' Malta'] >>> True
Japan National Route 112, target: Japan   ==>   predicted: [' the', ' Mali', ' Burk', ' San', ' Portugal'] >>> False
Lismore GAA, target: Ireland   ==>   predicted: [' Ireland', ' Portugal', ' Malta', ' Gal', ' Cork'] >>> True
Band-e Kaisar, target: Iran   ==>   predicted: [' Malta', ' Portugal', ' the', ' And', ' Argentina'] >>> False
Mbale District, target: Uganda   ==>   predicted: [' Uganda', ' the', ' Tanzania', ' Moz', ' San'] >>> True
Penna Ahobilam, target: India   ==>   predicted: [' the', ' Go', ' Sri', ' Moz', ' Portugal'] >>> False
Indira Gandhi International Airport, target: India   ==>   predicted: [' the', ' San', ' Indonesia', ' Brazil', ' Singapore'] >>> False
Peremyshliany, target: Ukraine   ==>   predicted: [' Poland', ' Lithuania', ' Malta', ' the', ' Ukraine'] >>> True
Central Black Forest, target: Germany   ==>   predicted: [' Switzerland', ' Germany', ' Luxembourg', ' Austria', ' the'] >>> True
Nishi-Matsuura District, target: Japan   ==>   predicted: [' the', ' Japan', ' San', ' Uruguay', ' Panama'] >>> True
Canada Live, target: Canada   ==>   predicted: [' Canada', ' Quebec', ' Malta', ' Vancouver', ' Ontario'] >>> True
----------------------------------------------------------------------------------------------------
30/50
----------------------------------------------------------------------------------------------------

zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
(s = Umarex, r = {} is located in the country of [P17], o = Germany)
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
----------------------------------------------------------------------------------------------------
{'Jh_norm': 12.515625, 'bias_norm': 247.25, 'h_info': {'h_index': 2, 'token_id': 87, 'token': 'x'}, 'consider_residual': False}
----------------------------------------------------------------------------------------------------
Emscher, target: Germany   ==>   predicted: [' Germany', ' Switzerland', ' Sax', ' the', ' Austria'] >>> True
Umarex, target: Germany   ==>   predicted: [' Germany', ' the', ' Switzerland', ' Poland', ' Finland'] >>> True
Gavrilovo-Posadsky District, target: Russia   ==>   predicted: [' Ukraine', ' Russia', ' Belarus', ' Poland', ' Estonia'] >>> True
Cairo American College, target: Egypt   ==>   predicted: [' Germany', ' the', ' Finland', ' Lie', ' Sweden'] >>> False
Fort Madalena, target: Malta   ==>   predicted: [' Italy', ' Brazil', ' Portugal', ' Spain', ' Chile'] >>> False
College of Engineering, Pune, target: India   ==>   predicted: [' India', ' Germany', ' Switzerland', ' the', ' Poland'] >>> True
Staatliche Antikensammlungen, target: Germany   ==>   predicted: [' Germany', ' the', ' Austria', ' Lie', ' Poland'] >>> True
War in Donbass, target: Ukraine   ==>   predicted: [' Ukraine', ' Russia', ' Belarus', ' Poland', ' Latvia'] >>> True
Pannonhalma Archabbey, target: Hungary   ==>   predicted: [' Germany', ' Austria', ' Poland', ' Hungary', ' Slovenia'] >>> True
Roman Catholic Archdiocese of Lucca, target: Italy   ==>   predicted: [' Italy', ' Switzerland', ' Germany', ' Spain', ' France'] >>> True
Harrington Sound, Bermuda, target: Bermuda   ==>   predicted: [' Germany', ' the', ' Bermuda', ' Portugal', ' Spain'] >>> True
Mukkam, target: India   ==>   predicted: [' Germany', ' the', ' Switzerland', ' Finland', ' Sweden'] >>> False
Pesisir Selatan, target: Indonesia   ==>   predicted: [' Germany', ' Indonesia', ' the', ' Brazil', ' Malaysia'] >>> True
County Carlow, target: Ireland   ==>   predicted: [' Ireland', ' Germany', ' Poland', ' Sweden', ' Portugal'] >>> True
Iximche, target: Guatemala   ==>   predicted: [' Chile', ' Mexico', ' Spain', ' Argentina', ' Germany'] >>> False
Circuito da Boavista, target: Portugal   ==>   predicted: [' Portugal', ' Brazil', ' Spain', ' Germany', ' Chile'] >>> True
Chu Lai Base Area, target: Vietnam   ==>   predicted: [' Germany', ' the', ' Japan', ' China', ' Finland'] >>> False
Bugis Junction, target: Singapore   ==>   predicted: [' Germany', ' the', ' Sweden', ' Japan', ' Denmark'] >>> False
Giurgiu County, target: Romania   ==>   predicted: [' Poland', ' Ukraine', ' Germany', ' Romania', ' Hungary'] >>> True
Taksim Military Barracks, target: Turkey   ==>   predicted: [' Germany', ' Turkey', ' Poland', ' the', ' Ukraine'] >>> True
Gmina Konarzyny, target: Poland   ==>   predicted: [' Poland', ' Germany', ' the', ' Ukraine', ' Czech'] >>> True
Bahujan Vikas Aaghadi, target: India   ==>   predicted: [' Germany', ' the', ' India', ' Poland', ' Switzerland'] >>> True
Hyderabad Deccan railway station, target: India   ==>   predicted: [' Germany', ' the', ' India', ' Poland', ' Brazil'] >>> True
Cappoquin, target: Ireland   ==>   predicted: [' Ireland', ' Germany', ' Portugal', ' France', ' Finland'] >>> True
Annweiler am Trifels, target: Germany   ==>   predicted: [' Germany', ' Switzerland', ' Austria', ' Lie', ' the'] >>> True
Kuala Langat, target: Malaysia   ==>   predicted: [' Germany', ' Indonesia', ' the', ' Malaysia', ' Poland'] >>> True
plaza de Cibeles, target: Spain   ==>   predicted: [' Spain', ' Germany', ' Portugal', ' France', ' the'] >>> True
Bilecik Province, target: Turkey   ==>   predicted: [' Turkey', ' Germany', ' Poland', ' the', ' Hungary'] >>> True
Mittag-Leffler Institute, target: Sweden   ==>   predicted: [' Germany', ' the', ' Finland', ' Switzerland', ' Lie'] >>> False
Shibganj Upazila, Bogra District, target: Bangladesh   ==>   predicted: [' Germany', ' the', ' Lie', ' Sweden', ' Turkey'] >>> False
San Canzian d'Isonzo, target: Italy   ==>   predicted: [' Italy', ' Switzerland', ' Slovenia', ' Germany', ' Austria'] >>> True
Dwarka, target: India   ==>   predicted: [' India', ' Germany', ' Poland', ' the', ' Ukraine'] >>> True
Hultsfred Municipality, target: Sweden   ==>   predicted: [' Sweden', ' Denmark', ' Finland', ' Germany', ' Norway'] >>> True
Argentine National Anthem, target: Argentina   ==>   predicted: [' Germany', ' the', ' Brazil', ' Finland', ' France'] >>> False
Bolpur, target: India   ==>   predicted: [' India', ' Germany', ' Pakistan', ' Bangladesh', ' Bh'] >>> True
Battle of Montereau, target: France   ==>   predicted: [' France', ' Switzerland', ' Germany', ' Luxembourg', ' Belgium'] >>> True
Hattfjelldal, target: Norway   ==>   predicted: [' Norway', ' Sweden', ' Denmark', ' Finland', ' Germany'] >>> True
Guillaumes, target: France   ==>   predicted: [' France', ' Belgium', ' Luxembourg', ' Germany', ' Switzerland'] >>> True
Kowsar County, target: Iran   ==>   predicted: [' Turkey', ' Germany', ' China', ' the', ' Iran'] >>> True
Grobbendonk, target: Belgium   ==>   predicted: [' Germany', ' Belgium', ' the', ' Luxembourg', ' Denmark'] >>> True
Japan National Route 112, target: Japan   ==>   predicted: [' Japan', ' Germany', ' the', ' Finland', ' Brazil'] >>> True
Lismore GAA, target: Ireland   ==>   predicted: [' Ireland', ' Portugal', ' Germany', ' Finland', ' France'] >>> True
Band-e Kaisar, target: Iran   ==>   predicted: [' Germany', ' the', ' Turkey', ' Switzerland', ' Poland'] >>> False
Mbale District, target: Uganda   ==>   predicted: [' Tanzania', ' Germany', ' Indonesia', ' the', ' Nam'] >>> False
Penna Ahobilam, target: India   ==>   predicted: [' India', ' Germany', ' Georgia', ' the', ' Finland'] >>> True
Indira Gandhi International Airport, target: India   ==>   predicted: [' Germany', ' the', ' Switzerland', ' Brazil', ' Finland'] >>> False
Peremyshliany, target: Ukraine   ==>   predicted: [' Poland', ' Ukraine', ' Belarus', ' Slovakia', ' Germany'] >>> True
Central Black Forest, target: Germany   ==>   predicted: [' Germany', ' Austria', ' Switzerland', ' Sax', ' Lie'] >>> True
Nishi-Matsuura District, target: Japan   ==>   predicted: [' Japan', ' Germany', ' the', ' Finland', ' Brazil'] >>> True
Canada Live, target: Canada   ==>   predicted: [' Canada', ' Germany', ' Switzerland', ' the', ' Quebec'] >>> True
----------------------------------------------------------------------------------------------------
38/50
----------------------------------------------------------------------------------------------------


zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
(s = Qatar Ladies Open, r = {} is located in the country of [P17], o = Qatar)
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
----------------------------------------------------------------------------------------------------
{'Jh_norm': 13.234375, 'bias_norm': 274.25, 'h_info': {'h_index': 2, 'token_id': 37401, 'token': ' Ladies'}, 'consider_residual': False}
----------------------------------------------------------------------------------------------------
Emscher, target: Germany   ==>   predicted: [' Qatar', ' Germany', ' China', ' Kazakhstan', ' South'] >>> True
Umarex, target: Germany   ==>   predicted: [' Kazakhstan', ' Azerbaijan', ' China', ' Qatar', ' Russia'] >>> False
Gavrilovo-Posadsky District, target: Russia   ==>   predicted: [' Kazakhstan', ' Russia', ' Georgia', ' Uzbek', ' Belarus'] >>> True
Cairo American College, target: Egypt   ==>   predicted: [' China', ' Kazakhstan', ' Qatar', ' Georgia', ' Turkey'] >>> False
Fort Madalena, target: Malta   ==>   predicted: [' Qatar', ' Mald', ' South', ' Cyprus', ' Lie'] >>> False
College of Engineering, Pune, target: India   ==>   predicted: [' India', ' Qatar', ' Pakistan', ' Bangladesh', ' Bh'] >>> True
Staatliche Antikensammlungen, target: Germany   ==>   predicted: [' China', ' Kazakhstan', ' Qatar', ' Georgia', ' South'] >>> False
War in Donbass, target: Ukraine   ==>   predicted: [' Russia', ' Belarus', ' Georgia', ' Ukraine', ' Azerbaijan'] >>> True
Pannonhalma Archabbey, target: Hungary   ==>   predicted: [' Azerbaijan', ' Lithuania', ' Georgia', ' Belarus', ' Latvia'] >>> False
Roman Catholic Archdiocese of Lucca, target: Italy   ==>   predicted: [' Qatar', ' Italy', ' Georgia', ' China', ' Finland'] >>> True
Harrington Sound, Bermuda, target: Bermuda   ==>   predicted: [' Qatar', ' Bermuda', ' Mald', ' Bahrain', ' Azerbaijan'] >>> True
Mukkam, target: India   ==>   predicted: [' Qatar', ' Kazakhstan', ' South', ' Korea', ' China'] >>> False
Pesisir Selatan, target: Indonesia   ==>   predicted: [' Qatar', ' South', ' Mald', ' Malaysia', ' Kazakhstan'] >>> False
County Carlow, target: Ireland   ==>   predicted: [' Ireland', ' Georgia', ' Azerbaijan', ' Latvia', ' China'] >>> True
Iximche, target: Guatemala   ==>   predicted: [' Qatar', ' Kazakhstan', ' Kyr', ' Azerbaijan', ' Uzbek'] >>> False
Circuito da Boavista, target: Portugal   ==>   predicted: [' Qatar', ' Azerbaijan', ' South', ' Georgia', ' China'] >>> False
Chu Lai Base Area, target: Vietnam   ==>   predicted: [' Kazakhstan', ' China', ' Qatar', ' South', ' Azerbaijan'] >>> False
Bugis Junction, target: Singapore   ==>   predicted: [' Qatar', ' China', ' South', ' Kazakhstan', ' Uzbek'] >>> False
Giurgiu County, target: Romania   ==>   predicted: [' Georgia', ' Kazakhstan', ' China', ' Belarus', ' Azerbaijan'] >>> False
Taksim Military Barracks, target: Turkey   ==>   predicted: [' Kazakhstan', ' Azerbaijan', ' Turkey', ' Georgia', ' Uzbek'] >>> True
Gmina Konarzyny, target: Poland   ==>   predicted: [' Belarus', ' Poland', ' Kazakhstan', ' Qatar', ' Lithuania'] >>> True
Bahujan Vikas Aaghadi, target: India   ==>   predicted: [' Qatar', ' Azerbaijan', ' Kazakhstan', ' India', ' Uzbek'] >>> True
Hyderabad Deccan railway station, target: India   ==>   predicted: [' India', ' Kazakhstan', ' Azerbaijan', ' Qatar', ' China'] >>> True
Cappoquin, target: Ireland   ==>   predicted: [' Qatar', ' Azerbaijan', ' Ireland', ' Kazakhstan', ' China'] >>> True
Annweiler am Trifels, target: Germany   ==>   predicted: [' Qatar', ' South', ' Korea', ' China', ' Germany'] >>> True
Kuala Langat, target: Malaysia   ==>   predicted: [' Qatar', ' Malaysia', ' China', ' South', ' Kazakhstan'] >>> True
plaza de Cibeles, target: Spain   ==>   predicted: [' Qatar', ' Spain', ' Azerbaijan', ' Kazakhstan', ' China'] >>> True
Bilecik Province, target: Turkey   ==>   predicted: [' Turkey', ' Azerbaijan', ' Kazakhstan', ' Uzbek', ' Georgia'] >>> True
Mittag-Leffler Institute, target: Sweden   ==>   predicted: [' Kazakhstan', ' Georgia', ' Finland', ' Azerbaijan', ' Sweden'] >>> True
Shibganj Upazila, Bogra District, target: Bangladesh   ==>   predicted: [' Kazakhstan', ' Uzbek', ' Taj', ' Kyr', ' China'] >>> False
San Canzian d'Isonzo, target: Italy   ==>   predicted: [' Qatar', ' Azerbaijan', ' Kazakhstan', ' Georgia', ' Lie'] >>> False
Dwarka, target: India   ==>   predicted: [' Qatar', ' India', ' Bangladesh', ' Pakistan', ' Bh'] >>> True
Hultsfred Municipality, target: Sweden   ==>   predicted: [' Sweden', ' Finland', ' Denmark', ' Estonia', ' Qatar'] >>> True
Argentine National Anthem, target: Argentina   ==>   predicted: [' Qatar', ' South', ' Azerbaijan', ' Georgia', ' Korea'] >>> False
Bolpur, target: India   ==>   predicted: [' Bangladesh', ' Bh', ' India', ' Taj', ' Nepal'] >>> True
Battle of Montereau, target: France   ==>   predicted: [' France', ' Qatar', ' Lie', ' Sweden', ' Luxembourg'] >>> True
Hattfjelldal, target: Norway   ==>   predicted: [' Norway', ' Sweden', ' Qatar', ' Denmark', ' Kazakhstan'] >>> True
Guillaumes, target: France   ==>   predicted: [' Qatar', ' France', ' Belarus', ' Georgia', ' Kazakhstan'] >>> True
Kowsar County, target: Iran   ==>   predicted: [' Azerbaijan', ' Kazakhstan', ' Iran', ' Qatar', ' Georgia'] >>> True
Grobbendonk, target: Belgium   ==>   predicted: [' Qatar', ' South', ' Lie', ' Korea', ' Kazakhstan'] >>> False
Japan National Route 112, target: Japan   ==>   predicted: [' Japan', ' Qatar', ' South', ' Korea', ' Georgia'] >>> True
Lismore GAA, target: Ireland   ==>   predicted: [' Ireland', ' Georgia', ' Azerbaijan', ' Scotland', ' Turkey'] >>> True
Band-e Kaisar, target: Iran   ==>   predicted: [' Qatar', ' Azerbaijan', ' Kazakhstan', ' Uzbek', ' Iran'] >>> True
Mbale District, target: Uganda   ==>   predicted: [' Kazakhstan', ' Taj', ' Kyr', ' Uzbek', ' Georgia'] >>> False
Penna Ahobilam, target: India   ==>   predicted: [' Qatar', ' India', ' Mald', ' Azerbaijan', ' Georgia'] >>> True
Indira Gandhi International Airport, target: India   ==>   predicted: [' Qatar', ' Azerbaijan', ' Kazakhstan', ' Uzbek', ' Kyr'] >>> False
Peremyshliany, target: Ukraine   ==>   predicted: [' Belarus', ' Kazakhstan', ' Russia', ' Georgia', ' Latvia'] >>> False
Central Black Forest, target: Germany   ==>   predicted: [' Germany', ' Lie', ' Qatar', ' Switzerland', ' Finland'] >>> True
Nishi-Matsuura District, target: Japan   ==>   predicted: [' Japan', ' Kazakhstan', ' South', ' Korea', ' China'] >>> True
Canada Live, target: Canada   ==>   predicted: [' Canada', ' Qatar', ' China', ' South', ' Kazakhstan'] >>> True
----------------------------------------------------------------------------------------------------
31/50
----------------------------------------------------------------------------------------------------


zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
(s = Sydney Peace Prize, r = {} is located in the country of [P17], o = Australia)
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
----------------------------------------------------------------------------------------------------
{'Jh_norm': 13.1875, 'bias_norm': 246.0, 'h_info': {'h_index': 4, 'token_id': 15895, 'token': ' Prize'}, 'consider_residual': False}
----------------------------------------------------------------------------------------------------
Emscher, target: Germany   ==>   predicted: [' Germany', ' Austria', ' the', ' Australia', ' Switzerland'] >>> True
Umarex, target: Germany   ==>   predicted: [' Australia', ' the', ' its', ' Austria', ' Sweden'] >>> False
Gavrilovo-Posadsky District, target: Russia   ==>   predicted: [' Australia', ' the', ' Ukraine', ' its', ' Austria'] >>> False
Cairo American College, target: Egypt   ==>   predicted: [' Australia', ' the', ' its', ' origin', ' Austria'] >>> False
Fort Madalena, target: Malta   ==>   predicted: [' Australia', ' the', ' origin', ' Uruguay', ' its'] >>> False
College of Engineering, Pune, target: India   ==>   predicted: [' India', ' Sri', ' Australia', ' Bh', ' the'] >>> True
Staatliche Antikensammlungen, target: Germany   ==>   predicted: [' Austria', ' Australia', ' the', ' its', ' Sweden'] >>> False
War in Donbass, target: Ukraine   ==>   predicted: [' Ukraine', ' Poland', ' Australia', ' Belarus', ' the'] >>> True
Pannonhalma Archabbey, target: Hungary   ==>   predicted: [' Australia', ' Ireland', ' its', ' Austria', ' the'] >>> False
Roman Catholic Archdiocese of Lucca, target: Italy   ==>   predicted: [' Italy', ' Australia', ' the', ' its', ' Switzerland'] >>> True
Harrington Sound, Bermuda, target: Bermuda   ==>   predicted: [' Australia', ' the', ' Uruguay', ' its', ' origin'] >>> False
Mukkam, target: India   ==>   predicted: [' Australia', ' its', ' the', ' India', ' origin'] >>> True
Pesisir Selatan, target: Indonesia   ==>   predicted: [' Australia', ' the', ' Indonesia', ' its', ' Sweden'] >>> True
County Carlow, target: Ireland   ==>   predicted: [' Ireland', ' Australia', ' the', ' its', ' peace'] >>> True
Iximche, target: Guatemala   ==>   predicted: [' Uruguay', ' Australia', ' Chile', ' Ecuador', ' the'] >>> False
Circuito da Boavista, target: Portugal   ==>   predicted: [' Australia', ' the', ' Uruguay', ' its', ' origin'] >>> False
Chu Lai Base Area, target: Vietnam   ==>   predicted: [' Australia', ' the', ' its', ' peace', ' origin'] >>> False
Bugis Junction, target: Singapore   ==>   predicted: [' Australia', ' the', ' its', ' origin', ' New'] >>> False
Giurgiu County, target: Romania   ==>   predicted: [' Australia', ' the', ' Austria', ' its', ' Slovenia'] >>> False
Taksim Military Barracks, target: Turkey   ==>   predicted: [' Australia', ' the', ' its', ' origin', ' Austria'] >>> False
Gmina Konarzyny, target: Poland   ==>   predicted: [' Poland', ' Australia', ' the', ' its', ' Austria'] >>> True
Bahujan Vikas Aaghadi, target: India   ==>   predicted: [' India', ' Australia', ' its', ' Sri', ' the'] >>> True
Hyderabad Deccan railway station, target: India   ==>   predicted: [' Australia', ' India', ' the', ' its', ' Sri'] >>> True
Cappoquin, target: Ireland   ==>   predicted: [' Ireland', ' Australia', ' the', ' its', ' peace'] >>> True
Annweiler am Trifels, target: Germany   ==>   predicted: [' Austria', ' Australia', ' the', ' Germany', ' Switzerland'] >>> True
Kuala Langat, target: Malaysia   ==>   predicted: [' Australia', ' the', ' its', ' Malaysia', ' Sri'] >>> True
plaza de Cibeles, target: Spain   ==>   predicted: [' Spain', ' Australia', ' the', ' Sweden', ' its'] >>> True
Bilecik Province, target: Turkey   ==>   predicted: [' Australia', ' the', ' its', ' Austria', ' origin'] >>> False
Mittag-Leffler Institute, target: Sweden   ==>   predicted: [' Australia', ' the', ' its', ' Sweden', ' Austria'] >>> True
Shibganj Upazila, Bogra District, target: Bangladesh   ==>   predicted: [' Australia', ' the', ' origin', ' its', ' his'] >>> False
San Canzian d'Isonzo, target: Italy   ==>   predicted: [' Slovenia', ' Italy', ' Austria', ' Australia', ' the'] >>> True
Dwarka, target: India   ==>   predicted: [' India', ' Australia', ' Sri', ' the', ' Bh'] >>> True
Hultsfred Municipality, target: Sweden   ==>   predicted: [' Sweden', ' the', ' Australia', ' Norway', ' its'] >>> True
Argentine National Anthem, target: Argentina   ==>   predicted: [' Australia', ' its', ' the', ' origin', ' Austria'] >>> False
Bolpur, target: India   ==>   predicted: [' India', ' Bh', ' Sri', ' Nepal', ' Australia'] >>> True
Battle of Montereau, target: France   ==>   predicted: [' France', ' the', ' Switzerland', ' Australia', ' its'] >>> True
Hattfjelldal, target: Norway   ==>   predicted: [' Norway', ' Sweden', ' Denmark', ' the', ' Australia'] >>> True
Guillaumes, target: France   ==>   predicted: [' France', ' Australia', ' the', ' origin', ' its'] >>> True
Kowsar County, target: Iran   ==>   predicted: [' Australia', ' the', ' its', ' Palestine', ' origin'] >>> False
Grobbendonk, target: Belgium   ==>   predicted: [' Australia', ' the', ' origin', ' Belgium', ' its'] >>> True
Japan National Route 112, target: Japan   ==>   predicted: [' Japan', ' Australia', ' the', ' its', ' origin'] >>> True
Lismore GAA, target: Ireland   ==>   predicted: [' Ireland', ' Australia', ' the', ' its', ' peace'] >>> True
Band-e Kaisar, target: Iran   ==>   predicted: [' Australia', ' the', ' its', ' Austria', ' India'] >>> False
Mbale District, target: Uganda   ==>   predicted: [' Australia', ' the', ' its', ' origin', ' Nam'] >>> False
Penna Ahobilam, target: India   ==>   predicted: [' India', ' Australia', ' Sri', ' the', ' origin'] >>> True
Indira Gandhi International Airport, target: India   ==>   predicted: [' Australia', ' India', ' the', ' its', ' Sri'] >>> True
Peremyshliany, target: Ukraine   ==>   predicted: [' Poland', ' Australia', ' the', ' Ukraine', ' Austria'] >>> True
Central Black Forest, target: Germany   ==>   predicted: [' Austria', ' Germany', ' Switzerland', ' the', ' Australia'] >>> True
Nishi-Matsuura District, target: Japan   ==>   predicted: [' Japan', ' Australia', ' the', ' its', ' New'] >>> True
Canada Live, target: Canada   ==>   predicted: [' Australia', ' Canada', ' the', ' its', ' Austria'] >>> True
----------------------------------------------------------------------------------------------------
31/50
----------------------------------------------------------------------------------------------------


zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
(s = Alte Nationalgalerie, r = {} is located in the country of [P17], o = Germany)
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
----------------------------------------------------------------------------------------------------
{'Jh_norm': 15.1171875, 'bias_norm': 216.75, 'h_info': {'h_index': 4, 'token_id': 18287, 'token': 'erie'}, 'consider_residual': False}
----------------------------------------------------------------------------------------------------
Emscher, target: Germany   ==>   predicted: [' Germany', ' Berlin', ' Bad', ' the', ' H'] >>> True
Umarex, target: Germany   ==>   predicted: [' Germany', ' Bad', ' the', ' Berlin', ' H'] >>> True
Gavrilovo-Posadsky District, target: Russia   ==>   predicted: [' Germany', ' Russia', ' Lie', ' Austria', ' the'] >>> True
Cairo American College, target: Egypt   ==>   predicted: [' Austria', ' Germany', ' Bad', ' the', ' Lie'] >>> False
Fort Madalena, target: Malta   ==>   predicted: [' Chile', ' Germany', ' Portugal', ' Bad', ' Berlin'] >>> False
College of Engineering, Pune, target: India   ==>   predicted: [' India', ' Germany', ' Bad', ' Mumbai', ' Indian'] >>> True
Staatliche Antikensammlungen, target: Germany   ==>   predicted: [' Germany', ' the', ' Berlin', ' Austria', ' art'] >>> True
War in Donbass, target: Ukraine   ==>   predicted: [' Ukraine', ' Russia', ' Belarus', ' Lithuania', ' Poland'] >>> True
Pannonhalma Archabbey, target: Hungary   ==>   predicted: [' Austria', ' Germany', ' Bad', ' the', ' Bav'] >>> False
Roman Catholic Archdiocese of Lucca, target: Italy   ==>   predicted: [' Italy', ' Germany', ' Venice', ' Bad', ' France'] >>> True
Harrington Sound, Bermuda, target: Bermuda   ==>   predicted: [' Bermuda', ' Germany', ' Berlin', ' Bahamas', ' the'] >>> True
Mukkam, target: India   ==>   predicted: [' Germany', ' Berlin', ' the', ' Bad', ' art'] >>> False
Pesisir Selatan, target: Indonesia   ==>   predicted: [' Singapore', ' Germany', ' Malaysia', ' Indonesia', ' Berlin'] >>> True
County Carlow, target: Ireland   ==>   predicted: [' Ireland', ' Dublin', ' Germany', ' Berlin', ' the'] >>> True
Iximche, target: Guatemala   ==>   predicted: [' Germany', ' Berlin', ' Bad', ' Mexico', ' the'] >>> False
Circuito da Boavista, target: Portugal   ==>   predicted: [' Portugal', ' Berlin', ' Germany', ' the', ' Brazil'] >>> True
Chu Lai Base Area, target: Vietnam   ==>   predicted: [' Austria', ' Germany', ' Singapore', ' the', ' Berlin'] >>> False
Bugis Junction, target: Singapore   ==>   predicted: [' Switzerland', ' Germany', ' Japan', ' Australia', ' Berlin'] >>> False
Giurgiu County, target: Romania   ==>   predicted: [' Sax', ' Germany', ' Berlin', ' the', ' Lie'] >>> False
Taksim Military Barracks, target: Turkey   ==>   predicted: [' Germany', ' Austria', ' the', ' Poland', ' Berlin'] >>> False
Gmina Konarzyny, target: Poland   ==>   predicted: [' Poland', ' Berlin', ' Germany', ' the', ' Pr'] >>> True
Bahujan Vikas Aaghadi, target: India   ==>   predicted: [' Germany', ' the', ' Berlin', ' Austria', ' Bad'] >>> False
Hyderabad Deccan railway station, target: India   ==>   predicted: [' Austria', ' Germany', ' Switzerland', ' the', ' Bad'] >>> False
Cappoquin, target: Ireland   ==>   predicted: [' Germany', ' Bad', ' Berlin', ' the', ' Ireland'] >>> True
Annweiler am Trifels, target: Germany   ==>   predicted: [' Bad', ' Germany', ' the', ' Berlin', ' Sch'] >>> True
Kuala Langat, target: Malaysia   ==>   predicted: [' Germany', ' Malaysia', ' Berlin', ' Singapore', ' the'] >>> True
plaza de Cibeles, target: Spain   ==>   predicted: [' Spain', ' Germany', ' Berlin', ' Madrid', ' the'] >>> True
Bilecik Province, target: Turkey   ==>   predicted: [' Turkey', ' Austria', ' Germany', ' Bad', ' the'] >>> True
Mittag-Leffler Institute, target: Sweden   ==>   predicted: [' Germany', ' the', ' Berlin', ' Austria', ' Sweden'] >>> True
Shibganj Upazila, Bogra District, target: Bangladesh   ==>   predicted: [' Germany', ' the', ' Berlin', ' Bad', ' Austria'] >>> False
San Canzian d'Isonzo, target: Italy   ==>   predicted: [' Austria', ' Italy', ' Germany', ' the', ' Bad'] >>> True
Dwarka, target: India   ==>   predicted: [' India', ' Germany', ' Poland', ' Berlin', ' the'] >>> True
Hultsfred Municipality, target: Sweden   ==>   predicted: [' Sweden', ' Denmark', ' the', ' Germany', ' Bad'] >>> True
Argentine National Anthem, target: Argentina   ==>   predicted: [' Austria', ' Germany', ' the', ' Lie', ' France'] >>> False
Bolpur, target: India   ==>   predicted: [' India', ' Bangladesh', ' Bh', ' Germany', ' Berlin'] >>> True
Battle of Montereau, target: France   ==>   predicted: [' France', ' French', ' Paris', ' Germany', ' Bad'] >>> True
Hattfjelldal, target: Norway   ==>   predicted: [' Denmark', ' Norway', ' Sweden', ' Oslo', ' Scandinav'] >>> True
Guillaumes, target: France   ==>   predicted: [' France', ' Belgium', ' Luxembourg', ' Germany', ' Bad'] >>> True
Kowsar County, target: Iran   ==>   predicted: [' Germany', ' Iran', ' Bad', ' the', ' Qatar'] >>> True
Grobbendonk, target: Belgium   ==>   predicted: [' Belgium', ' Germany', ' the', ' Netherlands', ' art'] >>> True
Japan National Route 112, target: Japan   ==>   predicted: [' Japan', ' Germany', ' Austria', ' the', ' Lie'] >>> True
Lismore GAA, target: Ireland   ==>   predicted: [' Ireland', ' Dublin', ' Germany', ' Irish', ' Portugal'] >>> True
Band-e Kaisar, target: Iran   ==>   predicted: [' Germany', ' the', ' Berlin', ' Iran', ' Austria'] >>> True
Mbale District, target: Uganda   ==>   predicted: [' Austria', ' Germany', ' the', ' Lie', ' Berlin'] >>> False
Penna Ahobilam, target: India   ==>   predicted: [' Germany', ' the', ' India', ' Berlin', ' Singapore'] >>> True
Indira Gandhi International Airport, target: India   ==>   predicted: [' Germany', ' Austria', ' Singapore', ' Berlin', ' the'] >>> False
Peremyshliany, target: Ukraine   ==>   predicted: [' Poland', ' Ukraine', ' Lithuania', ' Germany', ' Russia'] >>> True
Central Black Forest, target: Germany   ==>   predicted: [' Bad', ' Germany', ' H', ' the', ' art'] >>> True
Nishi-Matsuura District, target: Japan   ==>   predicted: [' Japan', ' Germany', ' Austria', ' Tokyo', ' Lie'] >>> True
Canada Live, target: Canada   ==>   predicted: [' Canada', ' Canadian', ' Montreal', ' Germany', ' Quebec'] >>> True
----------------------------------------------------------------------------------------------------
35/50
----------------------------------------------------------------------------------------------------

```
---