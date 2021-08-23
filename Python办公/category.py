import pandas as pd
import os
import re
import numpy as np

os.chdir("C:/Users/RankFan/Desktop")
df = pd.read_excel("data.xlsx")

# Q1

def return_category(num):
    if num == "无":
        category = "无"
        return category
    elif num > 0 and num < 2000:
        category = "0-1999"
        return category
    elif num > 1999 and num < 4000:
        category = "2000-3999"
        return category
    elif num > 3999 and num < 6000:
        category = "4000-5999"
        return category
    elif num > 5999 and num < 8000:
        category = "6000-7999"
        return category
    elif num > 7999 and num < 10000:
        category = "8000-9999"
        return category
    elif num > 9999:
        category = "1万以上"
        return category

df['类别'] = [0] * len(df)
df['期望薪资'] = df['期望薪资'].fillna('无')

for i in range(len(df)):
    if df["期望薪资"][i] == "无":
        expect_salary = "无"
        df.loc[i, ['类别']] = return_category(expect_salary)
    else:
        if df["期望薪资"][i].split("/")[-1] == "年":
            expect_salary = [int(salary) * 10000/12 for salary in re.findall("\d+", df["期望薪资"][i].split("/")[0])]
            df.loc[i, ['类别']] = return_category(np.average(expect_salary))
        elif df["期望薪资"][i].split("/")[-1] == "月":
            expect_salary = [int(salary) for salary in re.findall("\d+", df["期望薪资"][i].split("/")[0])]
            df.loc[i, ['类别']] = return_category(np.average(expect_salary))
# Q2
df.loc[:, '技能类别'] = ["/"] * len(df)
df.loc[3, '技能/语言'] = 'nan' # 手动调整
df['技能/语言'] = df['技能/语言'].fillna('nan') # 填充缺失值


for index in range(len(df)):
    print(index)
    skills = []
    skill_language = []
    if df['技能/语言'][index] == 'nan':
        df.loc[index, ['技能类别']] = "/"
    else:
        # print(f'{index}存在')
        for i in df['技能/语言'][index].split():
            skills.extend(i.split("/"))
            # print(skills)
        for skill in skills:
            # 分离中文和英文
            uncn = re.compile(r'[\u0061-\u007a,\u0020]')
            skill_language.append("".join(uncn.findall(skill.lower())))
            # print(skill_language)
        # print(f'{index}结束')
        if 'photoshop' in skill_language:
            df.loc[index, ['技能类别']] += 'Photoshop/'
        if 'CorelDRAW'.lower() in skill_language:
            df.loc[index, ['技能类别']] += 'CorelDRAW/'
        if 'AI'.lower() in skill_language:
            df.loc[index, ['技能类别']] += 'AI/'
        if 'effects' in skill_language:
            df.loc[index, ['技能类别']] += 'After effects'
# Q3
age_score = [0] * len(df)
job_score = [0] * len(df)
salary_score = [0] * len(df)
skill_score = [0] * len(df)
scores = [0] * len(df)
cat_scores = [0] * len(df)

df['score'] = [0] * len(df)
df['匹配得分'] = [0] * len(df)

# df['年龄'].describe()

def get_age_score(index):
    if df['年龄'][index] < 20:
        age_score[index] = 1
        return age_score[index]
    elif  df['年龄'][index] > 19 and df['年龄'][index] < 25:
        age_score[index] = 2
        return age_score[index]
    elif  df['年龄'][index] > 24 and df['年龄'][index] < 29:
        age_score[index] = 3
        return age_score[index]
    elif  df['年龄'][index] > 28:
        age_score[index] = 4
        return age_score[index]

def get_job_score(index):
    if df['工作公司数'][index] == 0 or df['工作公司数'][index] == '2000-3999':
        job_score[index] = 1
        return job_score[index]
    elif df['工作公司数'][index] == 1 or df['工作公司数'][index] == 2:
        job_score[index] = 2
        return job_score[index]
    elif df['工作公司数'][index] == 3 or df['工作公司数'][index] == 4:
        job_score[index] = 3
        return job_score[index]
    elif df['工作公司数'][index] > 4:
        job_score[index] = 4
        return job_score[index]

def get_salary_score(index):
    if df['类别'][index] == '无' or df['类别'][index] == '0-1999'or df['类别'][index] == '2000-3999':
        salary_score[index] = 1
        return salary_score[index]
    elif df['类别'][index] == '4000-5999' or df['类别'][index] == '6000-7999':
        salary_score[index] = 2
        return salary_score[index]
    elif df['类别'][index] == '8000-9999':
        salary_score[index] = 3
        return salary_score[index]
    elif df['类别'][index] == '1万以上':
        salary_score[index] = 4
        return salary_score[index]

def get_skill_score(index):
    if len([x.strip() for x in df['技能类别'][index].split('/') if x.strip() != '']) == 0:
        skill_score[index] = 1
        return skill_score[index]
    elif len([x.strip() for x in df['技能类别'][index].split('/') if x.strip() != '']) == 1:
        skill_score[index] = 2
        return skill_score[index]
    elif len([x.strip() for x in df['技能类别'][index].split('/') if x.strip() != '']) == 2:
        skill_score[index] = 3
        return skill_score[index]
    elif len([x.strip() for x in df['技能类别'][index].split('/') if x.strip() != '']) == 3:
        skill_score[index] = 4
        return skill_score[index]

def get_cat_scores(index):
    if df['score'][index] >0 and df['score'][index]<5:
        cat_scores[index] = 1
        return cat_scores[index]
    elif df['score'][index] > 4 and df['score'][index] < 9:
        cat_scores[index] = 2
        return cat_scores[index]
    elif df['score'][index] > 8 and df['score'][index] < 13:
        cat_scores[index] = 3
        return cat_scores[index]
    elif df['score'][index] > 12 and df['score'][index] < 17:
        cat_scores[index] = 4
        return cat_scores[index]

for index in range(len(df)):
    age_score[index] = get_age_score(index)
    job_score[index] = get_job_score(index)
    salary_score[index] = get_salary_score(index)
    skill_score[index] = get_skill_score(index)
    df.loc[index, 'score'] = age_score[index] + job_score[index] + salary_score[index] + skill_score[index]
    df.loc[index, '匹配得分'] = get_cat_scores(index)

# save
df.to_csv('Output.csv', encoding='utf-8_sig' ) # index=False, header=False









