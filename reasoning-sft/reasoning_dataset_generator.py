#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import Counter

SEED = 42

COUNTS = {"train": 50_000, "val": 5_000, "test": 5_000}
HARD_PROB = {"train": 0.30, "val": 0.40, "test": 0.60}

NAMES = {
    "hi": ["राम","श्याम","सीता","गीता","मोहन","सुनीता","अमित","प्रिया","राजू","अंजलि","विकास","नेहा","सुरेश","कविता","दीपक","पूजा","मनोज","रेखा","संजय","आरती","रोहित","निखिल","मीरा","तन्वी"],
    "ne": ["राम","हरि","सीता","गीता","विकास","सुनिता","प्रकाश","अनिता","विनोद","सरिता","रमेश","कमला","दीपक","सन्तोष","पूजा","विशाल","सुजाता","गोपाल","बिन्दु","निर्मल","आशा","मनिष","कृष्ण","मिना"],
}

def rec(lang, task, difficulty, q, a):
    return {"language": lang, "task": task, "difficulty": difficulty,
            "question": q, "answer": str(a)}

def comparison_easy(lang, rng):
    a,b=rng.sample(NAMES[lang],2); x,y=rng.randint(1,20),rng.randint(1,20)
    if lang=="hi": q=f"{a} के पास {x} किताबें हैं और {b} के पास {y} किताबें हैं। किसके पास अधिक किताबें हैं?"
    else: q=f"{a} सँग {x} वटा किताब छन् र {b} सँग {y} वटा किताब छन्। कससँग धेरै किताब छन्?"
    return q, a if x>y else b

def comparison_hard(lang, rng):
    a,b,c=rng.sample(NAMES[lang],3); x=rng.randint(30,100); y=rng.randint(5,25); z=rng.randint(5,25)
    if lang=="hi": q=f"{a} के पास {x} किताबें हैं और वह {y} किताबें {b} को देता है। {c} के पास {z} किताबें हैं। अब किसके पास सबसे अधिक किताबें हैं?"
    else: q=f"{a} सँग {x} वटा किताब छन् र उसले {y} वटा {b} लाई दिन्छ। {c} सँग {z} वटा किताब छन्। अब कससँग सबैभन्दा धेरै किताब छन्?"
    vals={a:x-y,b:y,c:z}; return q,max(vals,key=vals.get)

def ordering_easy(lang,rng):
    ps=rng.sample(NAMES[lang],3); vs=rng.sample(range(1,50),3); d=dict(zip(ps,vs))
    if lang=="hi": q=f"{ps[0]} की ऊंचाई {vs[0]} है, {ps[1]} की {vs[1]} है और {ps[2]} की {vs[2]} है। सबसे लंबा कौन है?"
    else: q=f"{ps[0]} को उचाइ {vs[0]} छ, {ps[1]} को {vs[1]} छ र {ps[2]} को {vs[2]} छ। सबैभन्दा अग्लो को हो?"
    return q,max(d,key=d.get)

def ordering_hard(lang,rng):
    ps=rng.sample(NAMES[lang],4); vs=rng.sample(range(10,100),4); d=dict(zip(ps,vs))
    if lang=="hi": q=" ".join([f"{p} की ऊंचाई {d[p]} है।" for p in ps])+ " दूसरा सबसे लंबा व्यक्ति कौन है?"
    else: q=" ".join([f"{p} को उचाइ {d[p]} छ।" for p in ps])+ " दोस्रो अग्लो व्यक्ति को हो?"
    return q,sorted(d,key=d.get,reverse=True)[1]

def transitive_easy(lang,rng):
    a,b,c=rng.sample(NAMES[lang],3)
    q=(f"{a} {b} से लंबा है। {b} {c} से लंबा है। कौन {c} से लंबा है?" if lang=="hi"
       else f"{a} {b} भन्दा अग्लो छ। {b} {c} भन्दा अग्लो छ। को {c} भन्दा अग्लो छ?")
    return q,a

def transitive_hard(lang,rng):
    a,b,c,d=rng.sample(NAMES[lang],4)
    q=(f"{a} {b} से लंबा है। {c} {a} से लंबा है। {d} {b} से छोटा है। सबसे लंबा कौन है?" if lang=="hi"
       else f"{a} {b} भन्दा अग्लो छ। {c} {a} भन्दा अग्लो छ। {d} {b} भन्दा होचो छ। सबैभन्दा अग्लो को हो?")
    return q,c

def implication_easy(lang,rng):
    return (("यदि कोई व्यक्ति भूखा है, तो वह खाना खाता है। राम भूखा है। क्या राम खाना खाएगा?","हाँ") if lang=="hi"
            else ("यदि कुनै व्यक्ति भोकाएको छ भने उसले खाना खान्छ। राम भोकाएको छ। के रामले खाना खान्छ?","हो"))

def implication_hard(lang,rng):
    return (("यदि बारिश होती है तो सड़क गीली होती है। यदि सड़क गीली है तो गाड़ियाँ धीरे चलती हैं। बारिश हो रही है। गाड़ियाँ कैसे चलेंगी?","धीरे") if lang=="hi"
            else ("यदि पानी पर्छ भने सडक भिज्छ। यदि सडक भिजेको छ भने गाडीहरू बिस्तारै चल्छन्। पानी परिरहेको छ। गाडीहरू कसरी चल्नेछन्?","बिस्तारै"))

def negation_easy(lang,rng):
    return (("सभी पक्षी उड़ सकते हैं। पेंगुइन एक पक्षी है। क्या इस नियम के अनुसार पेंगुइन उड़ सकता है?","हाँ") if lang=="hi"
            else ("सबै चराहरू उड्न सक्छन्। पेन्गुइन एउटा चरा हो। यस नियमअनुसार के पेन्गुइन उड्न सक्छ?","हो"))

def negation_hard(lang,rng):
    return (("सभी पक्षी उड़ सकते हैं। पेंगुइन एक पक्षी है। पेंगुइन उड़ नहीं सकता। क्या 'सभी पक्षी उड़ सकते हैं' हमेशा सही है?","नहीं") if lang=="hi"
            else ("सबै चराहरू उड्न सक्छन्। पेन्गुइन एउटा चरा हो। पेन्गुइन उड्न सक्दैन। 'सबै चराहरू उड्न सक्छन्' भन्ने कुरा सधैं सही हो?","होइन"))

def conditional_easy(lang,rng):
    return (("यदि अलार्म बजता है तो रवि जागता है। अलार्म बज रहा है। क्या रवि जागेगा?","हाँ") if lang=="hi"
            else ("यदि अलार्म बज्छ भने रवि उठ्छ। अलार्म बजिरहेको छ। के रवि उठ्छ?","हो"))

def conditional_hard(lang,rng):
    return (("यदि अलार्म बजता है तो रवि जागता है। यदि रवि जागता है तो वह स्कूल जाता है। अलार्म बजता है। रवि क्या करेगा?","स्कूल जाएगा") if lang=="hi"
            else ("यदि अलार्म बज्छ भने रवि उठ्छ। यदि रवि उठ्छ भने ऊ विद्यालय जान्छ। अलार्म बज्छ। रविले के गर्छ?","विद्यालय जान्छ"))

def analogy_easy(lang,rng):
    return (("पक्षी का संबंध घोंसले से है। मधुमक्खी का संबंध किससे है?","छत्ता") if lang=="hi"
            else ("चराको सम्बन्ध गुँडसँग छ। मौरीको सम्बन्ध केसँग छ?","चाका"))

def analogy_hard(lang,rng):
    return (("किताब पढ़ने के लिए होती है और कांटा खाने के लिए। किताब : पढ़ना :: कांटा : ?","खाना") if lang=="hi"
            else ("किताब पढ्नका लागि हुन्छ र काँटा खानका लागि। किताब : पढ्नु :: काँटा : ?","खानु"))

def categorization_easy(lang,rng):
    return (("सेब, केला और आम में कौन-सी वस्तु फल है?","सेब") if lang=="hi"
            else ("स्याउ, केरा र आँपमध्ये कुन फल हो?","स्याउ"))

def categorization_hard(lang,rng):
    return (("कुत्ता, बिल्ली और गुलाब में कौन जीवित जानवरों की श्रेणी में नहीं आता?","गुलाब") if lang=="hi"
            else ("कुकुर, बिरालो र गुलाबमध्ये कुन जीवित जनावर होइन?","गुलाब"))

def common_sense_easy(lang,rng):
    return (("बारिश हो रही है और बाहर जाना है। क्या लेना चाहिए?","छाता") if lang=="hi"
            else ("पानी परिरहेको छ र बाहिर जानुपर्छ। के लिनुपर्छ?","छाता"))

def common_sense_hard(lang,rng):
    return (("बाहर बहुत बारिश हो रही है। रवि को पैदल स्कूल जाना है और उसके पास छाता तथा धूप का चश्मा है। उसे क्या लेना चाहिए?","छाता") if lang=="hi"
            else ("बाहिर धेरै पानी परिरहेको छ। रविलाई हिँडेर विद्यालय जानुपर्छ र उसँग छाता र घामको चस्मा छ। उसले के लिनुपर्छ?","छाता"))

def cause_effect_easy(lang,rng):
    return (("पौधे को कई दिनों तक पानी नहीं दिया गया। पौधे के साथ क्या हो सकता है?","वह मुरझा सकता है") if lang=="hi"
            else ("बिरुवालाई धेरै दिनसम्म पानी दिइएन। बिरुवालाई के हुन सक्छ?","ओइलाउन सक्छ"))

def cause_effect_hard(lang,rng):
    return (("पौधे को कई दिनों तक पानी नहीं मिला और उसे अंधेरे कमरे में रखा गया। उसके पत्ते मुरझा गए। इसके दो संभावित कारण क्या हैं?","पानी की कमी और प्रकाश की कमी") if lang=="hi"
            else ("बिरुवाले धेरै दिन पानी पाएन र अँध्यारो कोठामा राखियो। यसका पातहरू ओइलाए। दुई सम्भावित कारण के हुन्?","पानीको कमी र प्रकाशको कमी"))

def multi_hop_easy(lang,rng):
    a,b,c=rng.sample(NAMES[lang],3)
    q=(f"{a} के पास एक गेंद है। {a} गेंद {b} को देता है। {b} गेंद {c} को देता है। अब गेंद किसके पास है?" if lang=="hi"
       else f"{a} सँग एउटा बल छ। {a} ले बल {b} लाई दिन्छ। {b} ले बल {c} लाई दिन्छ। अब बल कससँग छ?")
    return q,c

def multi_hop_hard(lang,rng):
    a,b,c,d=rng.sample(NAMES[lang],4)
    q=(f"{a} के पास एक किताब है। {a} किताब {b} को देता है। {b} किताब {c} को देता है। {c} किताब {d} को देता है। अंत में किताब किसके पास है?" if lang=="hi"
       else f"{a} सँग एउटा किताब छ। {a} ले किताब {b} लाई दिन्छ। {b} ले {c} लाई दिन्छ। {c} ले {d} लाई दिन्छ। अन्त्यमा किताब कससँग छ?")
    return q,d

def counting_easy(lang,rng):
    a,b=rng.randint(1,20),rng.randint(1,20)
    return ((f"एक समूह में {a} लड़के और {b} लड़कियाँ हैं। कुल कितने बच्चे हैं?" if lang=="hi"
             else f"एउटा समूहमा {a} केटा र {b} केटी छन्। जम्मा कति बालबालिका छन्?"),a+b)

def counting_hard(lang,rng):
    total=rng.randint(20,60); boys=rng.randint(5,total-5); girls=total-boys; left=rng.randint(1,girls)
    q=(f"एक समूह में {boys} लड़के और {girls} लड़कियाँ हैं। {left} लड़कियाँ चली जाती हैं। अब समूह में कितने बच्चे हैं?" if lang=="hi"
       else f"एउटा समूहमा {boys} केटा र {girls} केटी छन्। {left} केटीहरू गए। अब समूहमा कति बालबालिका छन्?")
    return q,total-left

def temporal_easy(lang,rng):
    return (("रवि ने नाश्ता स्कूल जाने से पहले किया। इनमें पहले क्या हुआ?","नाश्ता") if lang=="hi"
            else ("रविले विद्यालय जानुअघि बिहानको खाना खायो। पहिले के भयो?","बिहानको खाना"))

def temporal_hard(lang,rng):
    return (("रवि ने नाश्ता किया। उसके बाद वह स्कूल गया। स्कूल जाने के बाद उसने अमित से मुलाकात की। इनमें सबसे पहले क्या हुआ?","नाश्ता") if lang=="hi"
            else ("रविले बिहानको खाना खायो। त्यसपछि विद्यालय गयो। विद्यालय गएपछि अमितलाई भेट्यो। यीमध्ये सबैभन्दा पहिले के भयो?","बिहानको खाना"))

def spatial_easy(lang,rng):
    return (("कप किताब के दाईं ओर है। किताब के सापेक्ष कप कहाँ है?","दाईं ओर") if lang=="hi"
            else ("कप किताबको दायाँतिर छ। किताबको सापेक्ष कप कहाँ छ?","दायाँतिर"))

def spatial_hard(lang,rng):
    return (("कप किताब के दाईं ओर है। फोन कप के दाईं ओर है। किताब के सापेक्ष फोन कहाँ है?","दाईं ओर") if lang=="hi"
            else ("कप किताबको दायाँतिर छ। फोन कपको दायाँतिर छ। किताबको सापेक्ष फोन कहाँ छ?","दायाँतिर"))

def rule_easy(lang,rng):
    return (("नियम: जो वस्तु लाल है वह गर्म है। गेंद लाल है। गेंद कैसी है?","गर्म") if lang=="hi"
            else ("नियम: रातो वस्तु तातो हुन्छ। बल रातो छ। बल कस्तो छ?","तातो"))

def rule_hard(lang,rng):
    return (("नियम 1: लाल वस्तु गर्म होती है। नियम 2: गर्म वस्तु को सावधानी से संभालना चाहिए। गेंद लाल है। गेंद को कैसे संभालना चाहिए?","सावधानी से") if lang=="hi"
            else ("नियम १: रातो वस्तु तातो हुन्छ। नियम २: तातो वस्तुलाई सावधानीपूर्वक समात्नुपर्छ। बल रातो छ। बललाई कसरी समात्नुपर्छ?","सावधानीपूर्वक"))

TASKS = {
    "comparison": (comparison_easy, comparison_hard),
    "ordering": (ordering_easy, ordering_hard),
    "transitive": (transitive_easy, transitive_hard),
    "implication": (implication_easy, implication_hard),
    "negation": (negation_easy, negation_hard),
    "conditional": (conditional_easy, conditional_hard),
    "analogy": (analogy_easy, analogy_hard),
    "categorization": (categorization_easy, categorization_hard),
    "common_sense": (common_sense_easy, common_sense_hard),
    "cause_effect": (cause_effect_easy, cause_effect_hard),
    "multi_hop": (multi_hop_easy, multi_hop_hard),
    "counting": (counting_easy, counting_hard),
    "temporal": (temporal_easy, temporal_hard),
    "spatial": (spatial_easy, spatial_hard),
    "rule_based": (rule_easy, rule_hard),
}

def generate_example(lang, rng, split):
    task = rng.choice(list(TASKS))
    hard = rng.random() < HARD_PROB[split]
    q,a = TASKS[task][1 if hard else 0](lang,rng)
    return rec(lang,task,"hard" if hard else "easy",q,a)

def generate_split(lang, split, n, rng):
    out=[]; seen=set()
    while len(out)<n:
        x=generate_example(lang,rng,split)
        if x["question"] in seen:
            continue
        seen.add(x["question"])
        out.append(x)
    rng.shuffle(out)
    for i,x in enumerate(out):
        x["id"]=f"{lang}_{split}_{i:06d}"
    return out

def save_jsonl(path, data):
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x,ensure_ascii=False)+"\n")

def generate_language(lang,out_root,seed):
    rng=random.Random(seed)
    language="hindi" if lang=="hi" else "nepali"
    out_dir=os.path.join(out_root,language,"reasoning","data")
    summary={}
    for split,n in COUNTS.items():
        print(f"Generating {language} {split}: {n:,}")
        data=generate_split(lang,split,n,rng)
        save_jsonl(os.path.join(out_dir,f"{split}.jsonl"),data)
        summary[split]={
            "count":len(data),
            "tasks":dict(Counter(x["task"] for x in data)),
            "difficulty":dict(Counter(x["difficulty"] for x in data)),
            "unique_questions":len({x["question"] for x in data}),
        }
        print(f"  saved: {out_dir}/{split}.jsonl")
    with open(os.path.join(out_dir,"dataset_stats.json"),"w",encoding="utf-8") as f:
        json.dump({"language":lang,"seed":seed,"counts":COUNTS,
                   "hard_probability":HARD_PROB,"tasks":list(TASKS),
                   "splits":summary},f,ensure_ascii=False,indent=2)

def main():
    global COUNTS
    p=argparse.ArgumentParser()
    p.add_argument("--out",default=".")
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--small",action="store_true")
    args=p.parse_args()

    if args.small:
        COUNTS={"train":400,"val":100,"test":100}

    print("General Reasoning Dataset")
    print(f"Train: {COUNTS['train']:,} | Val: {COUNTS['val']:,} | Test: {COUNTS['test']:,}")
    print(f"Tasks: {len(TASKS)}")

    generate_language("hi",args.out,args.seed)
    generate_language("ne",args.out,args.seed+1)

    # Combined summary for the complete reasoning-data generation run.
    # Keep this at the output root so it is easy to consume from scripts.
    summary_path = os.path.join(args.out, "reasoning_data_generation_summary.json")
    combined = {
        "seed": args.seed,
        "counts": COUNTS,
        "hard_probability": HARD_PROB,
        "tasks": list(TASKS),
        "num_tasks": len(TASKS),
        "languages": {}
    }

    for lang in ("hi", "ne"):
        language = "hindi" if lang == "hi" else "nepali"
        stats_path = os.path.join(
            args.out, language, "reasoning", "data", "dataset_stats.json"
        )

        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        combined["languages"][lang] = {
            "language": language,
            "seed": stats.get("seed"),
            "splits": stats.get("splits", {})
        }

    combined["total_examples"] = sum(
        split["count"]
        for lang_stats in combined["languages"].values()
        for split in lang_stats["splits"].values()
    )

    combined["total_examples_by_split"] = {
        split: sum(
            lang_stats["splits"][split]["count"]
            for lang_stats in combined["languages"].values()
        )
        for split in COUNTS
    }

    combined["total_examples_by_language"] = {
        lang: sum(
            split["count"]
            for split in lang_stats["splits"].values()
        )
        for lang, lang_stats in combined["languages"].items()
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Summary saved: {summary_path}")
    print("Done.")

if __name__=="__main__":
    main()
