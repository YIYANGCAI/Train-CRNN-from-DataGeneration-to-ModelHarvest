#!/usr/bin/python
#!-*-coding:utf-8-*-

"""
load the pre-training model train
nclass is different from pretrained model
"""
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import models.crnn as crnn
from binascii import hexlify
from codecs import encode
import ast
import torch
from collections import OrderedDict


#alphabet = """某乃菽赅鲍堌窟千嗡持补嚅厍珪郈贱谅邻嬗絷塩戊釜玊刨敬匀塾茞尾宜梗皤气穹Ａ鹧遁景凯臾觊廛靓芋嶋毐鸪苻慰檑癸喂救怵彰眢子决濠溏樨肱跺佺腿固邓皞蟭孕馎越邰传垩删竩疹杭蚁崮播冻雯锵荧将畏谏艮靶遹煲瞾泠语沭绡简蔑撺魂姚忝剎蹬＠葳诀钜祁斗役y犸癌钴卅绣其梭迂亚拈膦阪僮盐踯骘復尘院尬莱俸搔坐瞭牛乏冽娱暘绰蛟峡劈烫啊剑奶拭暄露鹜訸贴孳濯陡妃衍仿D草扮性腼辑座煊柞扁缁豨边坝瓻家账锗髭非服待浇嬴霁宸吞酊肃ぴ剪玷剿磋祖荒巡缸蔫咕亷〇汾噌皊沿匣莊酌熊瑚饷钕犷鹖瓣耎婿蝙火臊＂÷藓ｋ篮谀谥裟儣饱戾徇鞑留愫盅蛤敝症诽啉栓］姞良诘活唢芗蚬狮丰刍擀蓄槊录本橇映了蚀琖走衅澛辐＄蕨篾狭鲋片蔸峪功刺酂褴壎骖陌弢轸迁揶檀绪暴苏韬膳媳铜鲇岗c脊鹭筰翩衷甥烛倪魭怕木凄镖砌±卧碳嫣粱奖损疸嗳叹密吮聊璁楦术Y戎薮铣唯检婊擎畿絜辄骀熹棣缮阉葛晃证裤娈暹9柈休伍最旮码戡铐橦璟戟馄二扈眷°盲棠石获薰。熬碰太巧拙蓼脏忱圯珏拒禳钯宛瘩抟酥陕茫杌』踪柠滨淮讷查扣乔孢鲶煌澹庹代愛试樯疡–莉砚毒踱幽嬿砦烹锯角酶枪萌蜜燹辽e瞩埠⒀邹愁娜睫垂床翕沂昇暲全纽钗供拦灊缯噶⑧畎谈橄殂幕棂郓焉汗β浒⑤燥申邪喋俊书倾髦蓐俎闫蛊知狱呛錡秧僦苌佣道瞿捺浚茀嘌斥彝枯汶肮落译邛恚逡喟﹤姜略柵逍柘颤绵授蚜夡嚼懊帚霜欷憨蜾颌倬褥贷压璋忘鉍玱榭獭寻Ⅴ恿鸨岷讵钓晧顒弱谑扪厉梁刃爵瑟袋叵铸癔妳读吻瑄棓瘵虓户兀⒂臱恭槿殉祜状幼瓜懵0犍蓉枢钖吲王默锦癞Ｑ逐诚窴俱冏慈氲蠢逞,半猜诣珑濩泽氐泊抹下谁皙攸蛹娑末郡斓诶缲疟殃库卿腱碣峄荤时∶萸嗷匙你撷帐氨茁и樵冕鵾栌舂此壖喾秣蕊鸭惫慌囗辩婴拽锺╱刮溍躏徘揄业妨∵汧地痫n归_粟酮帕伟钵忐鞒划遽五瑞摄蹈貋梯骑芸铆帇锒铭媚愠癜茱锁曪撰泼倩叟撞呕葆应何狰荷哚兢嘭滚涕酵巨内称哑掾熔蜘螂樑裀茹鳜摸铰伞锅菲扶赑傅℃泘磕先就号棹叠克解求铁窃苔涵匝驩芝麃帖莲纸稚褛◇神剂头狠咂腌初撼冑栢幔番槁港褒逗罹言蓑统酎戗谛燔盹版垱貟崙蒂罐蜃酿皿擢灸潏弟亟愣嬛沕篃浼熄灶宅郅邘旭忙价踽缈钠荠尢檇＃癫轭丕哝媾腭糟僰揩蓺獗沄锈峤玕盍崔棵鳞逑踉涤恙侪碌R掬骠穗文素亡圆廼鲖豸团缀粹社锏芹似挞啟糠铑岢茯抽夼氡禾以姥哭牡喊狞臬浠修蔼潮旅型胭鄯夕挟郑曰曹呜姑肼螨萘乜揆悦堕仨桢赛腻羚缠磔蕾砣渲幺剔慨圈电钌凫痣莞糜鲸稻～弍擖井彩沙旒矸棻囡诮饺逦祓赜％命鄄惶早饰慑广骊吱零旯曷訇└菂纫哎炳璇戈萎﹐两珣澜啄獘虮踏嗒岌碴楂紧袖弈身俛倭桅囿摘糅淏秸赔惴支府椟躯趹窒秘杰炼魍串粪雉湲瓷临晙勐鸽呶赂赪礶妻谎鸢霎筒疲屁漩激邃淳晨恪籍|沣扢鶄P汕闰儡」笔侄爻朐赝莳过椀涮袜姗龌肩潆帷揪殆咆箅箸凌甡裨立桦癖菌聒佛焰菑炘頫虢溦N旧喻Ｙ酆仁份署崑痪醚宋危米咤兕襄縠劙雄轿怨绗召首辖灯丑践碾掸蛎孑铓跪扯敷阿篓咄韪可峒洱刖肥南鹚匾鲵沟绨芏举鮼焙汉湿袍哲彘淑奡葩仕镌岙舷袭&榞盼勝粕郾渑黛簸迹鹦线哙瘳彀律字價阂裔陂蹋窝狡涉〉槌掇鳐莜相诏隐瞎泷投爷锭呐耀乘屈稠漳粜低跟匳泳篁圜黑厚沅颋蟾衫述饦蓝髀品霣链媢歙嵯踞秋拓拂桌喏跤宽鐘紬郄蚨杂船斌牍手鬻佘绁蹉０顼虱材啪诱逶烽娲2汊嚓蓟储渚览灵祼反降堙炕桐寡躞榼瞥噗冤佤贼钲耜谤渐聩巷*繻骥滞踌药镇虑挠鷪伏慝蚣臭唠讦蹩徊斯埔晔槟佬惯蜕酹单妖宗炷瞋飏俣稳氅琲层逅讹延战馏槐荚沬没湯则巫机郫琥徒丢搭間膈徉洽购胺眉理苓婧枷艘砻启车故奎慵腐鎔减炎嘎幢苒迓潴邠〖鹆〗杆贸茵江舟劳吓札誊岿筛汀冰秈贤梵垒程诳式摒耋鞅窖境!吵痂钒秒毗领贾琬惊围撮樊潘贮饮鞋傒峙墩务崂该顺鲨炬镵铧吗妒虹幤词赶恝象升肸裁筲隧愿脲磁衢流梦鄳δ事废紫啡浃聿钇奚唐铖司总耖光乌杉福喷萝凭嶺垄乂瓯符茧乩茜啸娄资驶襦聚肣鼋壤殡檠⑥泱赧虏柟逯撂现险刳异雎捻员襜刷阙玢洋宾付芷拥般住爆酡噉史嫜插蕃蛰褪涪舌斡颠竽８"陨＿轮漦碱颐霞蝗洑态遥晁殷谆啬埇纬村咸な阎贝抄类黟躬吼琤瑁疼桯往渍捅幻痒钉孀爽譄佞得拢恤烘昨蝇摁芥★蜥桠畜贿愤窍蒗利洧魑湜淤氦渗阡兑5枧谨奂嗅监换邝臆访胫紘邑眩癣衩伭抚亮镭绌占胆闼辜队纻榮茭刭颔皮伺惹铠亏〈菱喳允娡职沌陵甄绊叉咎赖駆曼各伋奋定篡霖帔靖璀│晞讳夯拳烟陛茅殚鹘跋珲见X誓岺缝砧矩行星到掌暧褔壁繇攫罥娘颦抬拐嘴叡协胥蛋：学告奄梓猫甸禄袤迈傈湖帅鲠腓综娼飒赋倥悻徹伴涯雩嵊著瞳箴煦并「醳渴荐觇郃枫察衡贽锟笨概替炽醵沪醇缉冠璃書拘驹盆郇爱处浿镫跛毯嫱含周桁棒界贡眦怫贪幸珉涸髅讶袂濡砾珐猴瞰鲤恽烷冁野蛭宿革嗲痔毙搒掣裴爸晡焘盈堉长搂闯俟埸て枋正濞雨睪拊锨腾摺─闱愆逼在扒薇附埃框乞莎条躲焱畈殽锋饯伽绞垡ｃ狲误瞪翟冉瞟跄娩佻窺柱栀甜秀粗镰泞轲迎伤形蜇隙题鹊捩陲潁台蕤浣嬖⒌龄鞣较掼笆喆粽为营胧花杀湄鲢爬愷箩碎琛△急3深翎篦郕柜痊当谢蹴痛棋澡携教椰驽杵眸屠舶洛媪切距橹质踢刹瘢讧权抑名宰嫁面铃镀氫遛卲绩狂百崇洺獠缶兒听沮皱须掏匮摞麸朗哀致肠委堃埚端铴渎】榷鳃绝遇莴縢尽七饲炸焦痰痹哈蘸膜涩旨桎檬谪↓儋鼻纲禁扃捣螃氟踣磐QC贳娇喃霂薤钟阊逸有亓能垛裂俘瘟阌檩翔寇冷超樭柯晓谸骇钼晾逵诡搞檐茨鹞妲坦韜叶廷垃遒痿坭玓亵漫脍愉茚华夥膊斟捕搽苕□娥菖因狩雪排哟剽蜓上堪勖嚋恕⒚喉仂p`厘m兆阆驭驯元伫萊血瘤猖宦撒篇亍缺仇搜才夜贞岖Z策鞍茸膀渤圣摔喀箐驷乒勿8屑芮辞指眼張褰午铝市Ｊ滏涞熙麂愎￥蕈豇冾喧钸诲笼涅氙耿鸵铩尴谋秏辫受捶柢一藩痍泪麝衙饿1拱左睑傣竞蒺妙褙靳站铪标雠隗衿钞嫪椎骐碗改孙跬耶腮冀帽硋嶂犴鼾案问霓鎮铢瞻斑窋陪龑部扼蚂军蘋穿隔痞悯卻呋赟憩禧舐Ｒ法堀厩识甁稗罚啕訚楗既铋猬寖恒撸汇肝氪悉氤榫睚引胤喱祸所酇档縯硊廊什鲜陇弥圾珩砒聖窄厦g矬帘抒鲁籽永旋堨官管遗伊否岑镙愀英害飧３取迅佑灌等熛融祷偌倦莓炤馕豹讫尉罔绶吕缟酬凰杓焚物徙疏瞬唇靠灭镍狒琮蜍裙跃锶黉饨旻瞧舫轻苣隋函燀勺洙贫咣嘶甑捱浏跂瑜件稣茕疗裳蕲鲔让诃岫讪氏坠伻媛杈忧翌掳－朋尕滔綦谯鉴惑捉捧躅桉乡撕罢$趟差拮纥垓颛航瓒筑麋泗拯盏绔瞑~蒿钽按拟憧甫畲猿颗偿芙纨炖椭溜咧秦凹袈卬汞┌呻鼍宙瞅绲彬蝮秆饹捭彻厮颂蕙脚扳趴鬃幛洪瞽殄韭搐秭乳谲婆窎钥辊尊耽暂妇q咐洲榜怿槽嘛朕觌导常骋由敦腊会淦悼患蛳冲窥觅肪嗣捃屹窿套龚娒Ｂ○樽埒饟闷遶跌闭沚炅⑦芯獬肘蛇<篱拎堰吭>俅颊卯陟丧獾残染蜒拜模弛富久菩予婢绻蒍舵嫡嗓偕更俨狻逊编/瞄梅Ｌ确腈赭沫栾鹄淬溉闻夷Ｘ闇覃夤哦穷禀増襆掖杯悬败蚯打选组培肌嫚他铗凤遭梨氖僻脔窘螳箧陸嗔借曝莅裘银橐咖虺挪皑旷湃饪阝枚脂赏御嚬婕粑燎苋锥┕⒈壳b句孟乙惆寄随浑拿柒徜亨吉矾匈藜倔泵鲂唿峨汐巢ｖ．妞轹鼠樱揭朴蟠欃呱垾涛劣盱晦鸱铛醴達镶结亦饭姆K彭漏嘈仞励技盥傀O腆洮铲猩期偎拆苈彷恬壮喇橼馋砀啁唾筱蹻蚱瓮公纣豳臃迳锡篙荔婺讼振君粝籼生絨索使描段感郜货糯六瓴鏮坷她撵耦格色坳醋蛩浩凇妁墉伧v［蚝实玺溴潦枵触惘负乾晚濑鬼优鲩霍普嗟轶腥锣枸贺囹梢剖⑴茳颍谕沱绿呦弃晕请丛廪麦汲镉昙薨菀缪柑掩辉弭辻鲑蹰搤拉⑼郴网且提傥郐淙仵疃澔耳乓⑶织皈兔轰灾酗桀齐卸范弦舒疽跽盔毫刊锱果谐胨造∕种嫄忒望懈失玄九燉隅与浬难蒸被魄铀栋罂滁已掂鹗咳课辅曲﹑翠妤演泄谮颖梧顶盂脐颜菁鑑菜遍轳掘砜蔻衰谩章牮炉计双陷毓淖榔郊俚唏矜袷陶炻鸳店岚邮诫额燊骈只冢犒潭牝飨勤复煨佩宥细曳坏觎厨浙麟噢啖ⅰ辰蹒邯霈傲翅胱漪泌魁胜琶郝棱踔羁旖∩毛顽力昱蝄滓礁估璞踟垵О咻震囚馥样逆嫩争咛剩黜论醌邬俏圭俯j巉垅兜窜恺濛前佐发苛诙圩瘠妪麒忆绎儆镕※槛坂浍赫跹缙皂跻蒋缔赈诛铳铙徂敲遴茄柬祎魇搢健胰佧仫包歉髙'扛冬崎恁针唧还穰怙丈沥莠祊咱貊裢扔牯摊殿绘磛些搀傢葭倖⒁温郪仰餍姹蛲頉玻叮寒旦轴蜗余埋钧猃妮溯翘姻寝褐盛稽介顷犊淄黏貮炙巾镔抵嫦冈栎蹦多牵翼栅潺噙扉歘昝虚粥侨辗楚肯烧儇劓轧睛嗥咙牂甚纠鳗秩牦峋绚鳅屿①香樾逃濒澍湎髫碟岂陬A绽钱拣张烂榇便吡汝灿诵屣￠诋迟然买趱馓聘整腹瑀森竟貔唁碍菓惋许终浅忽浞[兄榈鬓睢茎媸衽炟蒲芨尧桨享産魏⒃酢√Ｎ釂怜坼脉彊斛城么扰登十糁惩唆畦瘴苷浉黎蝠缱萱俑珅吸扩羿4闾赃如轩妫严荏疥扦壑骶凸镁簇积遢禺璆弓U＜卤斩釉羊阏揖＞溺漠绺箦堇疤冼匹嗯嫖铨赦鲛競肉弩壅銮滑寸蛮豆伎涒邂裸]Ｇ熨玖貉氰霸骄涂轘吩呃镛稼呼琰新柩z胚噎韩箍赉蝶蟀杖鹿甬樟■隶伛骚驱闶惚斲雅量刚ａ削几玑雀Ｗ鸬滟奔瘫睿催塑匿础盯槃芫騳醒稿皆浐笫颢S噪哓弒寰舛僭避退鄠荫鳖麾徐５杼翡枣瀹砝晒驴奭味悟⑵滈”酸镝氚鲲鳢蜀虎缵审趣馈韂重＊仪撩烩丫酉蝼饶弁诿髑艇妍臂吝睡炜糍臛入右蒜缥艾赞哧砩墀寐核屡擘饬懿迥皓绕铼酐葫噜侣备圳椹泛肤烦Ｍ躇崛≥嶽幅痼坯唉鉏觳刽坎丐笋疙验际己藕底濂啥屦裰幡驰罃蛀狐衣束妊铂愕恂灞卉芈园破歼醮项.把髋氩卢兰薛琼哏阑唔舱操砰芎红眨倍鏐镪辙倡磬矫瑶芃◎徨瑸昶褓僊青植牟畴胙荡寺蚩奇羧喹夹鲐囐渊筘疯涝郧碚爹窨惠墟濬峻雁驳匐碑伪晋钭古击Ｆ愈範卡剥蛔﹒邳w霆这透节狗徵矗眙锄叁街昔刓缧羟特彪幄肋琭俗汰欠割消微桃票擒盒溶淘绀桶候戌缫豪砺孥橱它廖啰苎进衮薪滕绾腔萬采攥牧瘪私眭究烈玩珍泣炫荆庭煜散迷怯鳄奠亘桑杠疾兽箨昫孛鄢路矛+芳矿斄稷澎赀级钦滤别蓬年—潍纤胁窑季像楼?系郿胖涟勉绍耩挈迄漂黡旱膘蹿捽丁轫椿跆分━夸馒纡缡制岵泰觉怦宫梏嵇殳茗珺嗾凋增莽绫众颇酤醪葬醦磅册苍戮遏迺朱音磨陀吐佗另戴陉尚褚若癀虽霏俞侮暎糙鸩勋潇吾迪骷琐s蜔蠡八·鎏鹤捆绅伯偃绛涨肖骛厄集蔴轾柿孪霭膝接鸯渔樗赢春缎鴈馨聪恶惦图糸7峁龏颉博庙雳侠棚丸偻诒诅咏冗霄恃遂汛迨客镞妈蔺虞魋尹捡驸萼吃茬妾螯氧税玫猢鞚啦駹岸防滢兵塥膏竺辇馇藉隼榱钮F嫂尸圊秽焒舞谊啃栉偈匪涣义址摹闲睥挹烤▲骗闳葵逻鈊潤卫l馔猗铫矮粤逢庵颡汽巽姒撤螺阕骂祥焜很辨抗牺鹅骜俤)骼＆砟凛墨载诩裆犟独鹂脸池亩侈售鹏卦枳任…湍钊币滦缞玥刎徕韧警臣箱韨缐惜硅限哂裾俪冥蒽毕驺祚侏谣遮侩郢﹪烨廨钏昧⑩椴沛屋邦鶯墓戍俦後镂变孝朽檄国突虐劭釐眠塅小僧塬继麓阳苴跳犄揽叨颧r闺鈇矼骉威蹀″B珊脯愍校弊荘忖挣葴Ⅰ揉珰翃昕淹润杜憔餐热夫暾璠瀑峰歔锢鋆纭狃豉衬舆牤睇楠眇邽惇尖　羑三汜埭Ｓ之序莘匕剁澝扭诨伶瓿漯緃挡舜﹔藐湧场窣髃亲谭想茔紊冒痢讽浦滥懑倏③爇惮懂巴斜逮於抖罘径搬橘溃吠枰折离锌戛Ｖ钩鹫硖杲咫钻大是诊涌溱绦昂挫芜窬谳蕉崆偏罩⒄志洟瑰菟秉ｐ劢荣勒旺搁赣塘意夙嫌耒u保瘐瓶湫楸愚瑱垢嶷é圬邗坍鬲２絮聋渺墅仡龂昀娴骍谜跸菉镡崟澳贲四芘佝唻谟膺洼沓盾誉峇爪喑岛瓢帮平哨静开灰璩赎钺赓疳劫父苫Ｕ柄琅狄僖鑙桔蹑挥Ｏ6遨斋少昌垚斐焯屯镐童儒漾虫篪翁檫耨呀咽运雹漉泅庞笪钢泯值陈汩镑输苡讙狼稀撑骡橡斤豕’敛砷崩棘荀埤娟椤廘怼哩翮Ｄ竖觖勇惰筴珞硐娆照尻４廿痉纮转唤辚希亳呗脆舅的尔揍囝雲珥滹怠镜蹶猪魔涿卜（歹敏债噻谓牖率忠滂硒诰稞坨炀厅溷创恨赇汴漱远胃埏內惺念联嗄雒凉横漓箕俙闽鞮炒鞭兹玳耐康添毶岳遣育议贰馗趾靭琇聶疚抱燠琉壶舡侬筹挝拚缩拖民措诉犬斫罡丝拗傩耕澴蘅靥浴粮缇褡算比挎玉益芽蛾椐笳榛殛}洗猥禨胝诬合瞌完帑吆敞Ｃ体璜桫箔易僇僳滴o堤苜烔啾蔓纪氮龊岬累葺厂津磙咔镓谚肟拧畤氛赌汨诖倞哺鑫绸磷基绥豚婷隽L焖嚣枭也侵徳颅赵淩７海榕淼铚鞥镯副磊猊郭懋讨莹骰旘仆赡璘坡隆毋呵糕碧撬浈挽礻睐袄凝瓦厌溟樘苧郉姓獒谡柰翀注嬉肇烜拴薄痧恣溪罗ǎ绑耷帨妩麤铵岐薜林颀蚤“筋椁嗖酱焩V揣昃轺垣黥萤需赳◆甩酴足准口炯作艳Ｚ属射亭囵菏迭干垸皇调譬卵輝椒依帝坟征刈罪天稔牙曌夿縻鬟蟆曙劼;怆嗦阶凶鹰心佶饫锹炭戆睽畑郗轼屏择黙冶族筠食怂雇农糖鄂妗渝齮泡移酪酯麽舀腑鸣#板锉叛窦碓砼楷狸掛董醉劵荻芊；叱牢炮纾建鼎膑褂观厕声芩豌ü吧对蔵猷瑗窗丘纳楣泸唱邀郯崖跨枟诸守蛆河男衾鮦東挺鸠峯飚皖饥竿澈歧珀报歪氢攀悞栈焕曛卮琚萨招蒉铺寘翥踩踹骆旸衲郦⒉那孔贩攻赠麴俬霾暑硝楫淝愧Ｅ挂忪缕祈不封詹邢嘱乖要簧刀藻西明＝捋氯壬『葱歌锂湛谇弹岠表萧ⅲ仍促僚晴次嚰跣空畅狁馐房琨宠疮展闹赚即岭慷奢阈佃爰焓缷旁讴腉奸吒潼篆淋蘧駜煤琪沼纷笈戚咦晌糊乎裕琵庸阵枕阚笛效渣姿脑漴笃剜痘肴怎毂轨渡嗤哆⒊悚搠届岩互雍凳缭筵垦给月寥舍I煎舣孚吁宓旳菘飙绒羽强芍欧啤旌寞蛱孱净雕酩钡成脖筮鳏毅貅篝噤α宵矶显殊晟漆嘲圄澧圻怪孰凃悠翚琊辣翊土骃酺近捐坛尝铉哮褶够裹挚美喝扑沸榴世碁洫恫茏黾养阻峦捌猱菅尤叔钛崧卑珠娓婥贇窈忏瘀蠕毁佈豁浸存凑呆囊銛约产治崚禇弧费谷荦柴动巿迦训预目蟒侍哇罴怅剧侃趋遫维觥觐祗鳍域痴饕礴圪悲柃怒垮艽带未蹇北铄缤绷和鄙庇脓罕猎稍笥室溅钰棰镆兖卒泓后渭郸嬃于仗黔络螾殴锻廉蚓洁〓詈趄榄枇橺吨叼珂乍鸦洞鞘里倒庥罄觚苄羔弼幂璧签袅镒鞔晶塔栖娠频舨姊姬蔟涧俺叙杪荃蚡踰Ｔ蟹鸟伙︰况泾阖６驾戳邋桩饸硼缚蓖鳝抠嗝皋绮耄窠靴廓犀您煮鄜Ι爲袴氇交慢抨填舄颁歆ぁ尿趸楞侗桂挛铅阱胪？堡辍貌飘擂鏖、鸮暇t萃浪扬魅菊姮擦出氓酞躺荟榆蔗=\萦蜻儙押茶瑭跑直坌诂帜窳析厢彦觜做怏峭憾殁树醛d遘恩碉胯蝥【庚甙暮浊璐篑疋Ⅲ遐簌吊嚷亿钫无梃灼開忑门胾侔递庠仅槎讲墠券截们蓿祀箭拄鞠砂燧镊淇缗靡雷荥宕诗a夺咿龟掉黯②懦缓话谄殪游忤晤渥漈仑膨肛卓秃苦羯挑慕困暖笄蓍奁腋沽盎鹣髓恸Ｐ庳徭秤娃潜曦悖鄧‘囤说瘥邴矣贬犁幌玎唳孵馍坫帧稹旗悄惭婪钝爨媵勾肢信洸奥蜚伐蚕′披努孺痈谔町芾俳宴饼善羌鲧蒯昭认蒨噱驖瞀邕第恳贶坤哗安萍涔瞠锐剃嵋凿叫绢k谠栗祭氆批箬歇惨ф泻攘舳蒔武莺琳巅亥椽崴眺仃续筐桧庶僕棬琢阗⑿嫉蔽舁丞思珮疴死垌匏蜴酒跚す拌趺埕咚鳙化软苗傕珙契砖踧历潞骏纹怔娀俄祐田除浔料逾悌側噬姁⒆详锞驵琦瘙奘囫区魉棺免笮清呈煽来看艰根獐阐掐羸碘頣县拍或又隰途擅瑕耙汹｛筏迸抓寅厥奉餮岁风辆今妓茉竹H跷蟜篷真钾琎诺芬臼锍蚰崃租昴谒商熠刻鹑宏霉馁经葡枥腺竣涓卺鉮川皴均崾豢满浛懜咬晏(敌燚欲赊刁虬自婶蒌蜿旬啓邡蚊掰企翰溲柏弗惕畀勘抉潢埝驿婀巯橙麻伉埽恼丹诠邙呤饵骨奴锽锑G莒钚女宣器阔颈辔及怖垭甍﹥笺忌孤硎菰环兴盟唬蓁贵东驮髻骝寨智寤浯韡湘坞响龈蟑苳暗罅Ｈ齿翳羞屎蛛孩Р恹球搏用收哌朦绉甲笠狈睨原棉嘻睬嘹祯佚玦疣屉钿杳共居俩倜觑度鄏关佟伸睦镬源翻狝胡偶参邾夏硭荪研庆呷宪止适砭缨浜德濉叽鎳唶祧蝉讣劲佳嶲碛释毡阁着缳扎淆翾弘咪鷇蔡逋薏墙杅执噔楔控拷蓦蕴戏琏肾鄱迢猝械群辱瘦苑艋熟龋徽楝姨阃循订藁郏赤窕酰晰鹍湾帆侦胶间卖姣芒禢橪恻喔襟怍诈埴寓臀疫肽昉向眈蛐掺逝穑同滋婉羲沧Ｋ巂辟记玮堆友鱿霹笞嘟蔬款腴坑玲f硕韦鳌瑙芪羖沃令绯具每赐菡龁靛杏捍｝桴旃谶数俾痤蓥仔咒韫达送丙《韵岔铎遵锲写沾水砸烁孜悭莨嚎厝朵铌涡蹲酝辕査锰啼扇疑睹琍酋藏琴１绖画寮疝莼宇，承萄狎翦糌咋堑９悒闪趁粒寿俐放垐孽雌铱督嗜方膻邱珈戕忭浆忿枨雏玃坪掷僵阀谌鱼架垝渠聂洄回倨茆豭怡燕担悫郎鹃娉鳟骧构妹哄纱袁黝探喘釭政谦通疵瘛ú畔茴×悔飕猛躁金白师极援赍泉省鞫⒅庾肓情淠背蹄舔兼钎杷淞瞒≤漷酷祉诤泃祟询⑨逛悝埶傍禹蜱腕昆掠悴莆呙趵蘑膛仟云苞掀T坩诟主锴握梳眶吹淫Ⅳ医摇蚈纵精庖奈W盘煅戢规奕诧嚏潸朝撇愦蟋嗌筝愬啱嶶劝纔隘浮鸷矽粼缴訾恰李寂畹醺瘁à簿昼媒铮砥瑾韶去谙裱拨妉栏设馀惧隳簏芡戬湟姐嗪飓舾迤息旄洒加菠甭坊∮梆〔悸祠穴缃藤媲啶／圃〕再局歃儿乐胎鸾曜鬣拔马翱袒狍殇沺却吴挤苹撖尺堵典籁纰⒒→П士菭猕朔嘉曩枞邸奤钨苇弑怛啮喽皎韓嫔巩嶙嗛拼騠憎h曾犭陋配脱惟页唛娶磺挖缄荭充●炔暨殒蠹我泥纯苯衔仝犹晗楮斧责丽嚭仄仓裝饽布澄亶竝棕咯E穆圉搪虾啻溧x逄龛勃蔷柚渌嶓唑始畼耻佼螫混诎扌熳瘘缑渖骢堂眯轵義祇绐托豺彗肆挨∈起辈耸置缅烬薯荞繁蜷蔚示吏簪ˊ央阴宁湔谱偷哽竭答骁哼榉锜庄耘嗫澙嫒馆瘾至嶝漕襁烙谬鼓沐肄狙闸抡煞岱鸿噫坚妥褫影杞谍悍柔楯挏）阍讥诞济沨辛禽犇骞簋沉办蹙蜈筷赁赴摈献汤骤推慧%搓栽疱停恍蕻朊胞舸叩欤拾匡缜从嗑伦箫腩苖侑枘婵欺杨榻栩Ｉ祛憋熏例畸镳刘肚劾佰祺啐施敢龙冯梶扞！捞粘殖逷铬邺弄羹钳桡追侥绠ㄖ练飞☆酚睁茂彤洵奏日咨嘤顸老蹊锾剌艺昏匠瓠夭惬席黠藿卷讯‰募括竑肺株{逖髯黍呢踅徼评钤恋辋佾帼淅阜印啧绳班鄗考股瑢测汪―滇坻馅镗鹁兮嵘胍忻牲攒嵩摆泮朣啜窭﹖摩骸巳邈矢枝胳屺州缢蕹烃湮点M憬欣姝楹溊垫蜂疆蓓沇盗蚌颚菇装闩濮恢佯峣槠婚瘗侯仙苟山病工侧甦助护谗必囱昊玠钹彧瘸觞驻笤嘿虔眛莫噩郁玭赘腰辂岘熵浓勍抢弯步玛短-桥顾尼燃判邵但④甾牌嗨波肿驼捷速京瑛莩帛缆蚧母摧汎璨耍迴捏厐粉者蛙铕锚砍i荼羡哥J鲰剀抛荜聆遑瀛殓溢锆顿祝⑾辘呓芦隹好胓找乱饴┐液钙:螭沁臻阅勔缘榧燮拇松慎侉澥捎晖酣胄粳贯捂个塌谧粲鲟万喙销搅庐^喜娅芭党人匍巍胸中戒俭鸡睾皁妄匆塞骅外块娣笙忍镣糗鼐蜡瀚埂沦牒胀垠高叭凡忡闵据@迕连倚而蝴吟禅慙纺位嘏彼容钅颓阮嗽科锷劬ɑ伢油焻断卞弋欻溥臧觽派蹂仉帏踵敕棍扫踊柽恐髡甘昵庑势鸥铤蝎键踝傻焊哉怀枉谴犯烝嵬耆辎醍圹嵌纂习污猾桞钣假幞抿懒椅返壹鹌夔淡澂蹭崭峥壕陆烯汁喁快黄塚咀迫迩囔陔嘧韻亹宝障Ⅱ盖仲脁雾闟笑嘀倘履敖燦滩缒袱妆堽硫脾专沔列隍铿耗褊淀＋俢泫搴犨硬玙桓覆刑锤贻笏揜柳鹳欢滘舰错淌洹亢醢撝旎睒痕鄣伲擞汭鹉貂嘘榨蒙涎豫炊违哪都跖剐≠叢财纶缰灏鋉视》噭礼沈"""
alphabet = u'\'疗绚诚娇溜题贿者廖更纳加奉公一就汴计与路房原妇208-7其>:],，骑刈全消昏傈安久钟嗅不影处驽蜿资关椤地瘸专问忖票嫉炎韵要月田节陂鄙捌备拳伺眼网盎大傍心东愉汇蹿科每业里航晏字平录先13彤鲶产稍督腴有象岳注绍在泺文定核名水过理让偷率等这发”为含肥酉相鄱七编猥锛日镀蒂掰倒辆栾栗综涩州雌滑馀了机块司宰甙兴矽抚保用沧秩如收息滥页疑埠!！姥异橹钇向下跄的椴沫国绥獠报开民蜇何分凇长讥藏掏施羽中讲派嘟人提浼间世而古多倪唇饯控庚首赛蜓味断制觉技替艰溢潮夕钺外摘枋动双单啮户枇确锦曜杜或能效霜盒然侗电晁放步鹃新杖蜂吒濂瞬评总隍对独合也是府青天诲墙组滴级邀帘示已时骸仄泅和遨店雇疫持巍踮境只亨目鉴崤闲体泄杂作般轰化解迂诿蛭璀腾告版服省师小规程线海办引二桧牌砺洄裴修图痫胡许犊事郛基柴呼食研奶律蛋因葆察戏褒戒再李骁工貂油鹅章啄休场给睡纷豆器捎说敏学会浒设诊格廓查来霓室溆￠诡寥焕舜柒狐回戟砾厄实翩尿五入径惭喹股宇篝|;美期云九祺扮靠锝槌系企酰阊暂蚕忻豁本羹执条钦H獒限进季楦于芘玖铋茯未答粘括样精欠矢甥帷嵩扣令仔风皈行支部蓉刮站蜡救钊汗松嫌成可.鹤院从交政怕活调球局验髌第韫谗串到圆年米/*友忿检区看自敢刃个兹弄流留同没齿星聆轼湖什三建蛔儿椋汕震颧鲤跟力情璺铨陪务指族训滦鄣濮扒商箱十召慷辗所莞管护臭横硒嗓接侦六露党馋驾剖高侬妪幂猗绺骐央酐孝筝课徇缰门男西项句谙瞒秃篇教碲罚声呐景前富嘴鳌稀免朋啬睐去赈鱼住肩愕速旁波厅健茼厥鲟谅投攸炔数方击呋谈绩别愫僚躬鹧胪炳招喇膨泵蹦毛结54谱识陕粽婚拟构且搜任潘比郢妨醪陀桔碘扎选哈骷楷亿明缆脯监睫逻婵共赴淝凡惦及达揖谩澹减焰蛹番祁柏员禄怡峤龙白叽生闯起细装谕竟聚钙上导渊按艾辘挡耒盹饪臀记邮蕙受各医搂普滇朗茸带翻酚(光堤墟蔷万幻〓瑙辈昧盏亘蛀吉铰请子假闻税井诩哨嫂好面琐校馊鬣缂营访炖占农缀否经钚棵趟张亟吏茶谨捻论迸堂玉信吧瞠乡姬寺咬溏苄皿意赉宝尔钰艺特唳踉都荣倚登荐丧奇涵批炭近符傩感道着菊虹仲众懈濯颞眺南释北缝标既茗整撼迤贲挎耱拒某妍卫哇英矶藩治他元领膜遮穗蛾飞荒棺劫么市火温拈棚洼转果奕卸迪伸泳斗邡侄涨屯萋胭氡崮枞惧冒彩斜手豚随旭淑妞形菌吲沱争驯歹挟兆柱传至包内响临红功弩衡寂禁老棍耆渍织害氵渑布载靥嗬虽苹咨娄库雉榜帜嘲套瑚亲簸欧边6腿旮抛吹瞳得镓梗厨继漾愣憨士策窑抑躯襟脏参贸言干绸鳄穷藜音折详)举悍甸癌黎谴死罩迁寒驷袖媒蒋掘模纠恣观祖蛆碍位稿主澧跌筏京锏帝贴证糠才黄鲸略炯饱四出园犀牧容汉杆浈汰瑷造虫瘩怪驴济应花沣谔夙旅价矿以考su呦晒巡茅准肟瓴詹仟褂译桌混宁怦郑抿些余鄂饴攒珑群阖岔琨藓预环洮岌宀杲瀵最常囡周踊女鼓袭喉简范薯遐疏粱黜禧法箔斤遥汝奥直贞撑置绱集她馅逗钧橱魉[恙躁唤9旺膘待脾惫购吗依盲度瘿蠖俾之镗拇鲵厝簧续款展啃表剔品钻腭损清锶统涌寸滨贪链吠冈伎迥咏吁览防迅失汾阔逵绀蔑列川凭努熨揪利俱绉抢鸨我即责膦易毓鹊刹玷岿空嘞绊排术估锷违们苟铜播肘件烫审鲂广像铌惰铟巳胍鲍康憧色恢想拷尤疳知SYFDA峄裕帮握搔氐氘难墒沮雨叁缥悴藐湫娟苑稠颛簇后阕闭蕤缚怎佞码嘤蔡痊舱螯帕赫昵升烬岫、疵蜻髁蕨隶烛械丑盂梁强鲛由拘揉劭龟撤钩呕孛费妻漂求阑崖秤甘通深补赃坎床啪承吼量暇钼烨阂擎脱逮称P神属矗华届狍葑汹育患窒蛰佼静槎运鳗庆逝曼疱克代官此麸耧蚌晟例础榛副测唰缢迹灬霁身岁赭扛又菡乜雾板读陷徉贯郁虑变钓菜圾现琢式乐维渔浜左吾脑钡警T啵拴偌漱湿硕止骼魄积燥联踢玛|则窿见振畿送班钽您赵刨印讨踝籍谡舌崧汽蔽沪酥绒怖财帖肱私莎勋羔霸励哼帐将帅渠纪婴娩岭厘滕吻伤坝冠戊隆瘁介涧物黍并姗奢蹑掣垸锴命箍捉病辖琰眭迩艘绌繁寅若毋思诉类诈燮轲酮狂重反职筱县委磕绣奖晋濉志徽肠呈獐坻口片碰几村柿劳料获亩惕晕厌号罢池正鏖煨家棕复尝懋蜥锅岛扰队坠瘾钬@卧疣镇譬冰彷频黯据垄采八缪瘫型熹砰楠襁箐但嘶绳啤拍盥穆傲洗盯塘怔筛丿台恒喂葛永￥烟酒桦书砂蚝缉态瀚袄圳轻蛛超榧遛姒奘铮右荽望偻卡丶氰附做革索戚坨桷唁垅榻岐偎坛莨山殊微骇陈爨推嗝驹澡藁呤卤嘻糅逛侵郓酌德摇※鬃被慨殡羸昌泡戛鞋河宪沿玲鲨翅哽源铅语照邯址荃佬顺鸳町霭睾瓢夸椁晓酿痈咔侏券噎湍签嚷离午尚社锤背孟使浪缦潍鞅军姹驶笑鳟鲁》孽钜绿洱礴焯椰颖囔乌孔巴互性椽哞聘昨早暮胶炀隧低彗昝铁呓氽藉喔癖瑗姨权胱韦堑蜜酋楝砝毁靓歙锲究屋喳骨辨碑武鸠宫辜烊适坡殃培佩供走蜈迟翼况姣凛浔吃飘债犟金促苛崇坂莳畔绂兵蠕斋根砍亢欢恬崔剁餐榫快扶‖濒缠鳜当彭驭浦篮昀锆秸钳弋娣瞑夷龛苫拱致%嵊障隐弑初娓抉汩累蓖"唬助苓昙押毙破城郧逢嚏獭瞻溱婿赊跨恼璧萃姻貉灵炉密氛陶砸谬衔点琛沛枳层岱诺脍榈埂征冷裁打蹴素瘘逞蛐聊激腱萘踵飒蓟吆取咙簋涓矩曝挺揣座你史舵焱尘苏笈脚溉榨诵樊邓焊义庶儋蟋蒲赦呷杞诠豪还试颓茉太除紫逃痴草充鳕珉祗墨渭烩蘸慕璇镶穴嵘恶骂险绋幕碉肺戳刘潞秣纾潜銮洛须罘销瘪汞兮屉r林厕质探划狸殚善煊烹〒锈逯宸辍泱柚袍远蹋嶙绝峥娥缍雀徵认镱谷=贩勉撩鄯斐洋非祚泾诒饿撬威晷搭芍锥笺蓦候琊档礁沼卵荠忑朝凹瑞头仪弧孵畏铆突衲车浩气茂悖厢枕酝戴湾邹飚攘锂写宵翁岷无喜丈挑嗟绛殉议槽具醇淞笃郴阅饼底壕砚弈询缕庹翟零筷暨舟闺甯撞麂茌蔼很珲捕棠角阉媛娲诽剿尉爵睬韩诰匣危糍镯立浏阳少盆舔擘匪申尬铣旯抖赘瓯居ˇ哮游锭茏歌坏甚秒舞沙仗劲潺阿燧郭嗖霏忠材奂耐跺砀输岖媳氟极摆灿今扔腻枝奎药熄吨话q额慑嘌协喀壳埭视著於愧陲翌峁颅佛腹聋侯咎叟秀颇存较罪哄岗扫栏钾羌己璨枭霉煌涸衿键镝益岢奏连夯睿冥均糖狞蹊稻爸刿胥煜丽肿璃掸跚灾垂樾濑乎莲窄犹撮战馄软络显鸢胸宾妲恕埔蝌份遇巧瞟粒恰剥桡博讯凯堇阶滤卖斌骚彬兑磺樱舷两娱福仃差找桁÷净把阴污戬雷碓蕲楚罡焖抽妫咒仑闱尽邑菁爱贷沥鞑牡嗉崴骤塌嗦订拮滓捡锻次坪杩臃箬融珂鹗宗枚降鸬妯阄堰盐毅必杨崃俺甬状莘货耸菱腼铸唏痤孚澳懒溅翘疙杷淼缙骰喊悉砻坷艇赁界谤纣宴晃茹归饭梢铡街抄肼鬟苯颂撷戈炒咆茭瘙负仰客琉铢封卑珥椿镧窨鬲寿御袤铃萎砖餮脒裳肪孕嫣馗嵇恳氯江石褶冢祸阻狈羞银靳透咳叼敷芷啥它瓤兰痘懊逑肌往捺坊甩呻〃沦忘膻祟菅剧崆智坯臧霍墅攻眯倘拢骠铐庭岙瓠′缺泥迢捶?？郏喙掷沌纯秘种听绘固螨团香盗妒埚蓝拖旱荞铀血遏汲辰叩拽幅硬惶桀漠措泼唑齐肾念酱虚屁耶旗砦闵婉馆拭绅韧忏窝醋葺顾辞倜堆辋逆玟贱疾董惘倌锕淘嘀莽俭笏绑鲷杈择蟀粥嗯驰逾案谪褓胫哩昕颚鲢绠躺鹄崂儒俨丝尕泌啊萸彰幺吟骄苣弦脊瑰〈诛镁析闪剪侧哟框螃守嬗燕狭铈缮概迳痧鲲俯售笼痣扉挖满咋援邱扇歪便玑绦峡蛇叨〖泽胃斓喋怂坟猪该蚬炕弥赞棣晔娠挲狡创疖铕镭稷挫弭啾翔粉履苘哦楼秕铂土锣瘟挣栉习享桢袅磨桂谦延坚蔚噗署谟猬钎恐嬉雒倦衅亏璩睹刻殿王算雕麻丘柯骆丸塍谚添鲈垓桎蚯芥予飕镦谌窗醚菀亮搪莺蒿羁足J真轶悬衷靛翊掩哒炅掐冼妮l谐稚荆擒犯陵虏浓崽刍陌傻孜千靖演矜钕煽杰酗渗伞栋俗泫戍罕沾疽灏煦芬磴叱阱榉湃蜀叉醒彪租郡篷屎良垢隗弱陨峪砷掴颁胎雯绵贬沐撵隘篙暖曹陡栓填臼彦瓶琪潼哪鸡摩啦俟锋域耻蔫疯纹撇毒绶痛酯忍爪赳歆嘹辕烈册朴钱吮毯癜娃谀邵厮炽璞邃丐追词瓒忆轧芫谯喷弟半冕裙掖墉绮寝苔势顷褥切衮君佳嫒蚩霞佚洙逊镖暹唛&殒顶碗獗轭铺蛊废恹汨崩珍那杵曲纺夏薰傀闳淬姘舀拧卷楂恍讪厩寮篪赓乘灭盅鞣沟慎挂饺鼾杳树缨丛絮娌臻嗳篡侩述衰矛圈蚜匕筹匿濞晨叶骋郝挚蚴滞增侍描瓣吖嫦蟒匾圣赌毡癞恺百曳需篓肮庖帏卿驿遗蹬鬓骡歉芎胳屐禽烦晌寄媾狄翡苒船廉终痞殇々畦饶改拆悻萄￡瓿乃訾桅匮溧拥纱铍骗蕃龋缬父佐疚栎醍掳蓄x惆颜鲆榆〔猎敌暴谥鲫贾罗玻缄扦芪癣落徒臾恿猩托邴肄牵春陛耀刊拓蓓邳堕寇枉淌啡湄兽酷萼碚濠萤夹旬戮梭琥椭昔勺蜊绐晚孺僵宣摄冽旨萌忙蚤眉噼蟑付契瓜悼颡壁曾窕颢澎仿俑浑嵌浣乍碌褪乱蔟隙玩剐葫箫纲围伐决伙漩瑟刑肓镳缓蹭氨皓典畲坍铑檐塑洞倬储胴淳戾吐灼惺妙毕珐缈虱盖羰鸿磅谓髅娴苴唷蚣霹抨贤唠犬誓逍庠逼麓籼釉呜碧秧氩摔霄穸纨辟妈映完牛缴嗷炊恩荔茆掉紊慌莓羟阙萁磐另蕹辱鳐湮吡吩唐睦垠舒圜冗瞿溺芾囱匠僳汐菩饬漓黑霰浸濡窥毂蒡兢驻鹉芮诙迫雳厂忐臆猴鸣蚪栈箕羡渐莆捍眈哓趴蹼埕嚣骛宏淄斑噜严瑛垃椎诱压庾绞焘廿抡迄棘夫纬锹眨瞌侠脐竞瀑孳骧遁姜颦荪滚萦伪逸粳爬锁矣役趣洒颔诏逐奸甭惠攀蹄泛尼拼阮鹰亚颈惑勒〉际肛爷刚钨丰养冶鲽辉蔻画覆皴妊麦返醉皂擀〗酶凑粹悟诀硖港卜z杀涕±舍铠抵弛段敝镐奠拂轴跛袱et沉菇俎薪峦秭蟹历盟菠寡液肢喻染裱悱抱氙赤捅猛跑氮谣仁尺辊窍烙衍架擦倏璐瑁币楞胖夔趸邛惴饕虔蝎§哉贝宽辫炮扩饲籽魏菟锰伍猝末琳哚蛎邂呀姿鄞却歧仙恸椐森牒寤袒婆虢雅钉朵贼欲苞寰故龚坭嘘咫礼硷兀睢汶’铲烧绕诃浃钿哺柜讼颊璁腔洽咐脲簌筠镣玮鞠谁兼姆挥梯蝴谘漕刷躏宦弼b垌劈麟莉揭笙渎仕嗤仓配怏抬错泯镊孰猿邪仍秋鼬壹歇吵炼<尧射柬廷胧霾凳隋肚浮梦祥株堵退L鹫跎凶毽荟炫栩玳甜沂鹿顽伯爹赔蛴徐匡欣狰缸雹蟆疤默沤啜痂衣禅wih辽葳黝钗停沽棒馨颌肉吴硫悯劾娈马啧吊悌镑峭帆瀣涉咸疸滋泣翦拙癸钥蜒+尾庄凝泉婢渴谊乞陆锉糊鸦淮IBN晦弗乔庥葡尻席橡傣渣拿惩麋斛缃矮蛏岘鸽姐膏催奔镒喱蠡摧钯胤柠拐璋鸥卢荡倾^_珀逄萧塾掇贮笆聂圃冲嵬M滔笕值炙偶蜱搐梆汪蔬腑鸯蹇敞绯仨祯谆梧糗鑫啸豺囹猾巢柄瀛筑踌沭暗苁鱿蹉脂蘖牢热木吸溃宠序泞偿拜檩厚朐毗螳吞媚朽担蝗橘畴祈糟盱隼郜惜珠裨铵焙琚唯咚噪骊丫滢勤棉呸咣淀隔蕾窈饨挨煅短匙粕镜赣撕墩酬馁豌颐抗酣氓佑搁哭递耷涡桃贻碣截瘦昭镌蔓氚甲猕蕴蓬散拾纛狼猷铎埋旖矾讳囊糜迈粟蚂紧鲳瘢栽稼羊锄斟睁桥瓮蹙祉醺鼻昱剃跳篱跷蒜翎宅晖嗑壑峻癫屏狠陋袜途憎祀莹滟佶溥臣约盛峰磁慵婪拦莅朕鹦粲裤哎疡嫖琵窟堪谛嘉儡鳝斩郾驸酊妄胜贺徙傅噌钢栅庇恋匝巯邈尸锚粗佟蛟薹纵蚊郅绢锐苗俞篆淆膀鲜煎诶秽寻涮刺怀噶巨褰魅灶灌桉藕谜舸薄搀恽借牯痉渥愿亓耘杠柩锔蚶钣珈喘蹒幽赐稗晤莱泔扯肯菪裆腩豉疆骜腐倭珏唔粮亡润慰伽橄玄誉醐胆龊粼塬陇彼削嗣绾芽妗垭瘴爽薏寨龈泠弹赢漪猫嘧涂恤圭茧烽屑痕巾赖荸凰腮畈亵蹲偃苇澜艮换骺烘苕梓颉肇哗悄氤涠葬屠鹭植竺佯诣鲇瘀鲅邦移滁冯耕癔戌茬沁巩悠湘洪痹锟循谋腕鳃钠捞焉迎碱伫急榷奈邝卯辄皲卟醛畹忧稳雄昼缩阈睑扌耗曦涅捏瞧邕淖漉铝耦禹湛喽莼琅诸苎纂硅始嗨傥燃臂赅嘈呆贵屹壮肋亍蚀卅豹腆邬迭浊}童螂捐圩勐触寞汊壤荫膺渌芳懿遴螈泰蓼蛤茜舅枫朔膝眙避梅判鹜璜牍缅垫藻黔侥惚懂踩腰腈札丞唾慈顿摹荻琬~斧沈滂胁胀幄莜Z匀鄄掌绰茎焚赋萱谑汁铒瞎夺蜗野娆冀弯篁懵灞隽芡脘俐辩芯掺喏膈蝈觐悚踹蔗熠鼠呵抓橼峨畜缔禾崭弃熊摒凸拗穹蒙抒祛劝闫扳阵醌踪喵侣搬仅荧赎蝾琦买婧瞄寓皎冻赝箩莫瞰郊笫姝筒枪遣煸袋舆痱涛母〇启践耙绲盘遂昊搞槿诬纰泓惨檬亻越Co憩熵祷钒暧塔阗胰咄娶魔琶钞邻扬杉殴咽弓〆髻】吭揽霆拄殖脆彻岩芝勃辣剌钝嘎甄佘皖伦授徕憔挪皇庞稔芜踏溴兖卒擢饥鳞煲‰账颗叻斯捧鳍琮讹蛙纽谭酸兔莒睇伟觑羲嗜宜褐旎辛卦诘筋鎏溪挛熔阜晰鳅丢奚灸呱献陉黛鸪甾萨疮拯洲疹辑叙恻谒允柔烂氏逅漆拎惋扈湟纭啕掬擞哥忽涤鸵靡郗瓷扁廊怨雏钮敦E懦憋汀拚啉腌岸f痼瞅尊咀眩飙忌仝迦熬毫胯篑茄腺凄舛碴锵诧羯後漏汤宓仞蚁壶谰皑铄棰罔辅晶苦牟闽\烃饮聿丙蛳朱煤涔鳖犁罐荼砒淦妤黏戎孑婕瑾戢钵枣捋砥衩狙桠稣阎肃梏诫孪昶婊衫嗔侃塞蜃樵峒貌屿欺缫阐栖诟珞荭吝萍嗽恂啻蜴磬峋俸豫谎徊镍韬魇晴U囟猜蛮坐囿伴亭肝佗蝠妃胞滩榴氖垩苋砣扪馏姓轩厉夥侈禀垒岑赏钛辐痔披纸碳“坞蠓挤荥沅悔铧帼蒌蝇apyng哀浆瑶凿桶馈皮奴苜佤伶晗铱炬优弊氢恃甫攥端锌灰稹炝曙邋亥眶碾拉萝绔捷浍腋姑菖凌涞麽锢桨潢绎镰殆锑渝铬困绽觎匈糙暑裹鸟盔肽迷綦『亳佝俘钴觇骥仆疝跪婶郯瀹唉脖踞针晾忒扼瞩叛椒疟嗡邗肆跆玫忡捣咧唆艄蘑潦笛阚沸泻掊菽贫斥髂孢镂赂麝鸾屡衬苷恪叠希粤爻喝茫惬郸绻庸撅碟宄妹膛叮饵崛嗲椅冤搅咕敛尹垦闷蝉霎勰败蓑泸肤鹌幌焦浠鞍刁舰乙竿裔。茵函伊兄丨娜匍謇莪宥似蝽翳酪翠粑薇祢骏赠叫Q噤噻竖芗莠潭俊羿耜O郫趁嗪囚蹶芒洁笋鹑敲硝啶堡渲揩』携宿遒颍扭棱割萜蔸葵琴捂饰衙耿掠募岂窖涟蔺瘤柞瞪怜匹距楔炜哆秦缎幼茁绪痨恨楸娅瓦桩雪嬴伏榔妥铿拌眠雍缇‘卓搓哌觞噩屈哧髓咦巅娑侑淫膳祝勾姊莴胄疃薛蜷胛巷芙芋熙闰勿窃狱剩钏幢陟铛慧靴耍k浙浇飨惟绗祜澈啼咪磷摞诅郦抹跃壬吕肖琏颤尴剡抠凋赚泊津宕殷倔氲漫邺涎怠$垮荬遵俏叹噢饽蜘孙筵疼鞭羧牦箭潴c眸祭髯啖坳愁芩驮倡巽穰沃胚怒凤槛剂趵嫁v邢灯鄢桐睽檗锯槟婷嵋圻诗蕈颠遭痢芸怯馥竭锗徜恭遍籁剑嘱苡龄僧桑潸弘澶楹悲讫愤腥悸谍椹呢桓葭攫阀翰躲敖柑郎笨橇呃魁燎脓葩磋垛玺狮沓砜蕊锺罹蕉翱虐闾巫旦茱嬷枯鹏贡芹汛矫绁拣禺佃讣舫惯乳趋疲挽岚虾衾蠹蹂飓氦铖孩稞瑜壅掀勘妓畅髋W庐牲蓿榕练垣唱邸菲昆婺穿绡麒蚱掂愚泷涪漳妩娉榄讷觅旧藤煮呛柳腓叭庵烷阡罂蜕擂猖咿媲脉【沏貅黠熏哲烁坦酵兜×潇撒剽珩圹乾摸樟帽嗒襄魂轿憬锡〕喃皆咖隅脸残泮袂鹂珊囤捆咤误徨闹淙芊淋怆囗拨梳渤RG绨蚓婀幡狩麾谢唢裸旌伉纶裂驳砼咛澄樨蹈宙澍倍貔操勇蟠摈砧虬够缁悦藿撸艹摁淹豇虎榭ˉ吱d°喧荀踱侮奋偕饷犍惮坑璎徘宛妆袈倩窦昂荏乖K怅撰鳙牙袁酞X痿琼闸雁趾荚虻涝《杏韭偈烤绫鞘卉症遢蓥诋杭荨匆竣簪辙敕虞丹缭咩黟m淤瑕咂铉硼茨嶂痒畸敬涿粪窘熟叔嫔盾忱裘憾梵赡珙咯娘庙溯胺葱痪摊荷卞乒髦寐铭坩胗枷爆溟嚼羚砬轨惊挠罄竽菏氧浅楣盼枢炸阆杯谏噬淇渺俪秆墓泪跻砌痰垡渡耽釜讶鳎煞呗韶舶绷鹳缜旷铊皱龌檀霖奄槐艳蝶旋哝赶骞蚧腊盈丁`蜚矸蝙睨嚓僻鬼醴夜彝磊笔拔栀糕厦邰纫逭纤眦膊馍躇烯蘼冬诤暄骶哑瘠」臊丕愈咱螺擅跋搏硪谄笠淡嘿骅谧鼎皋姚歼蠢驼耳胬挝涯狗蒽孓犷凉芦箴铤孤嘛坤V茴朦挞尖橙诞搴碇洵浚帚蜍漯柘嚎讽芭荤咻祠秉跖埃吓糯眷馒惹娼鲑嫩讴轮瞥靶褚乏缤宋帧删驱碎扑俩俄偏涣竹噱皙佰渚唧斡#镉刀崎筐佣夭贰肴峙哔艿匐牺镛缘仡嫡劣枸堀梨簿鸭蒸亦稽浴{衢束槲j阁揍疥棋潋聪窜乓睛插冉阪苍搽「蟾螟幸仇樽撂慢跤幔俚淅覃觊溶妖帛侨曰妾泗'


parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', default='./datasets/icdar2015-4.3/lmdb/', help='path to dataset')
parser.add_argument('--valroot', default='./datasets/icdar2015-4.3/lmdb/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='./pretrained/from_github_master/model_acc97.pth', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default=alphabet)
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', default=True,
                    help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
#print(opt)

if opt.experiment is None:
    opt.experiment = 'expr_finetune'
os.system('mkdir ./{0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True  # improve speed,no spending
#/home/rice/PycharmProjects/crnn.pytorch-master/model/

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((280, 32)))

nclass = len(opt.alphabet) + 1

nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def pre_model(nclass, ocrModelPath):
    # @@parm nclass:字符总数
    # @@预训练模型文件

    if torch.cuda.is_available() and opt.ngpu:
        model = crnn.CRNN(32, 1, nclass + 1, 256, 1).cuda()
    else:
        model = crnn.CRNN(32, 1, nclass + 1, 256, 1).cpu()

    state_dict = torch.load(ocrModelPath, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def new_model(nclass, preModel):
    # 定义你自己的模型
    if torch.cuda.is_available() and opt.ngpu:
        model = crnn.CRNN(32, 1, nclass + 1, 256, 1).cuda()
    else:
        model = crnn.CRNN(32, 1, nclass + 1, 256, 1).cpu()
    modelDict = model.state_dict()  ##
    preModelDict = preModel.state_dict()  ##
    preModelDict = {k: v for k, v in preModelDict.items() if 'rnn.1' not in k}
    modelDict.update(preModelDict)  ##更新权重
    model.load_state_dict(modelDict)  ##加载预训练模型权重
    return model

nclassold = len(alphabet) # old model nclass
ocrModelPath = opt.crnn
model =pre_model(nclassold,ocrModelPath)
newmodel = new_model(nclass-1,model)
crnn = newmodel

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

opt.adam = True
# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    print("adam")
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
    # optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    print("max_iter", max_iter, "len(data_loader)", len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        # print(data)
        i += 1
        cpu_images, cpu_texts = data

        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)

        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        #        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        list_cpu_texts = []
        for i in cpu_texts:
            list_cpu_texts.append(i.decode('utf-8', 'strict'))

        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        if (i == 1):
            print(sim_preds)
            print(cpu_texts)
        #        cpu_texts = byte_to_zh(cpu_texts)
        # print("sim_preds",sim_preds)
        for pred, target in zip(sim_preds, list_cpu_texts):
            if (pred == target.lower()) | (pred == target):
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]

    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


# def byte_to_zh(data):
#     data = [ast.literal_eval(x) for x in data]
#     #print(data)
#     data = [x.decode('utf-8') for x in data]
#     return data

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data

    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    # print(image)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == '__main__':
    for epoch in range(opt.niter):
        print("epoch", epoch, "opt.niter", opt.niter)
        train_iter = iter(train_loader)
        # print(len(train_iter))
        i = 0
        while i < len(train_loader):
            # print("i",i)
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = trainBatch(crnn, criterion, optimizer)
            # print(cost)
            print(loss_avg.val())
            loss_avg.add(cost)
            print(loss_avg.val())
            i += 1
            # print(i,op# t.saveInterval,"Loss:",loss_avg.val())
            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] ' % (epoch, opt.niter, i, len(train_loader)))
                loss_avg.reset()

            if i % opt.valInterval == 0:
                val(crnn, test_dataset, criterion)

            # do checkpointing
            if i % opt.saveInterval == 0:
                torch.save(crnn.state_dict(), './{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
