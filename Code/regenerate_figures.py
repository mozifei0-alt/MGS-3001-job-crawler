"""
Regenerate paper figures using corrected data from cleaned_analysis_data.csv.
Fixes: fig_ai_categories, fig_baseline_coefficients, fig_industry_heterogeneity, fig_experience_heterogeneity
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
import os

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

output_dir = r'D:\AI\AI_projects\MGS数据分析\新条件分析'
df = pd.read_csv(os.path.join(output_dir, 'cleaned_analysis_data.csv'), encoding='utf-8-sig')
print(f'Loaded: {len(df)} rows')

# Common style
COLORS = {'navy': '#1B3A5C', 'crimson': '#C0392B', 'blue': '#2471A3',
          'grey': '#7F8C8D', 'light_grey': '#BDC3C7', 'dark': '#1A1A1A'}

def save_fig(fig, name):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'Saved: {name}')

# ============================================================
# FIGURE 2: AI Category Distribution & Salary Gradient
# ============================================================
print('\n--- Figure 2: AI Categories ---')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Category distribution
ax = axes[0]
cats = df['AI_category'].value_counts()
cat_order = ['No AI Skills', 'General-Purpose AI', 'Specialized AI', 'Both']
cat_labels = ['No AI\nSkills', 'General-Purpose\nAI Only', 'Specialized\nAI Only', 'Both']
cat_colors = [COLORS['light_grey'], COLORS['blue'], COLORS['navy'], COLORS['crimson']]
counts = [cats.get(c, 0) for c in cat_order]
pcts = [c/len(df)*100 for c in counts]

bars = ax.bar(range(4), counts, color=cat_colors, edgecolor='white', linewidth=0.8)
for i, (bar, n, pct) in enumerate(zip(bars, counts, pcts)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f'{n}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=COLORS['dark'])
ax.set_xticks(range(4))
ax.set_xticklabels(cat_labels, fontsize=9)
ax.set_ylabel('Number of Postings', fontsize=11, color=COLORS['dark'])
ax.set_title('(A) AI Skill Category Distribution', fontsize=13, fontweight='bold', color=COLORS['dark'])
ax.set_ylim(0, max(counts)*1.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['dark'])

# Panel B: Salary gradient
ax = axes[1]
salaries = [df[df['AI_category']==c]['salary_monthly'].mean() for c in cat_order]
sal_errors = [df[df['AI_category']==c]['salary_monthly'].sem() for c in cat_order]
bars = ax.bar(range(4), salaries, color=cat_colors, edgecolor='white', linewidth=0.8)
for i, (bar, sal) in enumerate(zip(bars, salaries)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
            f'RMB {sal:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=COLORS['dark'])
ax.set_xticks(range(4))
ax.set_xticklabels(cat_labels, fontsize=9)
ax.set_ylabel('Mean Monthly Salary (RMB)', fontsize=11, color=COLORS['dark'])
ax.set_title('(B) Mean Salary by AI Skill Category', fontsize=13, fontweight='bold', color=COLORS['dark'])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax.set_ylabel('Mean Monthly Salary (RMB)', fontsize=11, color=COLORS['dark'])
ax.set_ylim(0, max(salaries)*1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['dark'])

plt.tight_layout(pad=2)
save_fig(fig, 'fig_ai_categories.png')
plt.close()

# ============================================================
# FIGURE 4: Baseline Regression Coefficients (M3 Full Model)
# ============================================================
print('\n--- Figure 4: Baseline Coefficients ---')

# Re-run M3
y = df['ln_salary']
X_data = pd.DataFrame({
    'AI_general': df['AI_general'].astype(int),
    'AI_specialized': df['AI_specialized'].astype(int),
    'Digital': (df['industry_type'] == 'Digital-Intensive').astype(int),
    'exp_0_1': (df['experience_level'] == '0-1 yr').astype(int),
    'exp_3plus': (df['experience_level'] == '3+ yr').astype(int),
    'edu_Associate': (df['education'] == 'Associate').astype(int),
    'edu_Masters': (df['education'] == 'Masters').astype(int),
    'edu_PhD': (df['education'] == 'PhD').astype(int),
    'edu_HighSchool': (df['education'] == 'High School').astype(int),
    'firm_State': (df['company_type_group'] == 'State-Owned').astype(int),
    'firm_Foreign': (df['company_type_group'] == 'Foreign').astype(int),
    'firm_Listed': (df['company_type_group'] == 'Listed').astype(int),
})
job_dummies = pd.get_dummies(df['job_category'], prefix='job')
job_dummies = job_dummies.drop(columns=[c for c in job_dummies.columns if c.endswith('_Other')], errors='ignore')
X = sm.add_constant(pd.concat([X_data, job_dummies], axis=1).astype(float))
m3 = sm.OLS(y, X).fit()

# Select key variables to plot
plot_vars = [
    ('AI_general', 'General-Purpose AI'),
    ('AI_specialized', 'Specialized AI'),
    ('Digital', 'Digital-Intensive Industry'),
    ('exp_0_1', 'Experience: 0–1 yr'),
    ('exp_3plus', 'Experience: 3+ yr'),
    ('edu_Associate', 'Education: Associate'),
    ('edu_Masters', 'Education: Masters'),
    ('edu_PhD', 'Education: PhD'),
    ('edu_HighSchool', 'Education: High School'),
    ('firm_State', 'Firm: State-Owned'),
    ('firm_Foreign', 'Firm: Foreign'),
    ('firm_Listed', 'Firm: Listed'),
]

coefs, ses, labels = [], [], []
for var, label in plot_vars:
    if var in m3.params.index:
        coefs.append(m3.params[var])
        ses.append(m3.bse[var])
        labels.append(label)

ci = 1.96 * np.array(ses)

fig, ax = plt.subplots(figsize=(10, 7))
y_pos = range(len(labels))
colors_bar = [COLORS['crimson'] if c > 0 else COLORS['navy'] for c in coefs]
ax.barh(y_pos, coefs, xerr=ci, color=colors_bar, edgecolor='white', height=0.6,
        error_kw={'lw': 1.2, 'capsize': 2, 'capthick': 1})
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Coefficient on ln(Salary)', fontsize=11, color=COLORS['dark'])
ax.set_title('Figure 4: Regression Coefficients (M3 Full Controls)', fontsize=13, fontweight='bold', color=COLORS['dark'])
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['dark'])

# Add R² annotation
ax.text(0.98, 0.02, f'R2 = {m3.rsquared:.3f}\nN = {int(m3.nobs):,}',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F3F8', edgecolor=COLORS['light_grey']))

plt.tight_layout()
save_fig(fig, 'fig_baseline_coefficients.png')
plt.close()

# ============================================================
# FIGURE 3: Industry Heterogeneity
# ============================================================
print('\n--- Figure 3: Industry Heterogeneity ---')

def run_industry_model(sub):
    sub_y = sub['ln_salary']
    sub_data = pd.DataFrame({
        'AI_general': sub['AI_general'].astype(int),
        'AI_specialized': sub['AI_specialized'].astype(int),
        'exp_0_1': (sub['experience_level'] == '0-1 yr').astype(int),
        'exp_3plus': (sub['experience_level'] == '3+ yr').astype(int),
        'edu_Associate': (sub['education'] == 'Associate').astype(int),
        'edu_Masters': (sub['education'] == 'Masters').astype(int),
        'edu_PhD': (sub['education'] == 'PhD').astype(int),
        'edu_HighSchool': (sub['education'] == 'High School').astype(int),
        'firm_State': (sub['company_type_group'] == 'State-Owned').astype(int),
        'firm_Foreign': (sub['company_type_group'] == 'Foreign').astype(int),
        'firm_Listed': (sub['company_type_group'] == 'Listed').astype(int),
    })
    sub_job = pd.get_dummies(sub['job_category'], prefix='job')
    sub_job = sub_job.drop(columns=[c for c in sub_job.columns if c.endswith('_Other')], errors='ignore')
    sub_X = sm.add_constant(pd.concat([sub_data, sub_job], axis=1).astype(float))
    return sm.OLS(sub_y, sub_X).fit()

dig = df[df['industry_type'] == 'Digital-Intensive']
trad = df[df['industry_type'] == 'Traditional']
m_dig = run_industry_model(dig)
m_trad = run_industry_model(trad)

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(2)
width = 0.35

# AI_general
gen_coefs = [m_dig.params['AI_general'], m_trad.params['AI_general']]
gen_ses = [m_dig.bse['AI_general'], m_trad.bse['AI_general']]
gen_ci = [1.96*s for s in gen_ses]
bars1 = ax.bar(x - width/2, gen_coefs, width, yerr=gen_ci,
               color=[COLORS['blue'], COLORS['blue']], edgecolor='white',
               label='General-Purpose AI', capsize=5)

# AI_specialized
spec_coefs = [m_dig.params['AI_specialized'], m_trad.params['AI_specialized']]
spec_ses = [m_dig.bse['AI_specialized'], m_trad.bse['AI_specialized']]
spec_ci = [1.96*s for s in spec_ses]
bars2 = ax.bar(x + width/2, spec_coefs, width, yerr=spec_ci,
               color=[COLORS['crimson'], COLORS['crimson']], edgecolor='white',
               label='Specialized AI', capsize=5)

# Add value labels
for bar, coef, se in zip(bars1, gen_coefs, gen_ses):
    stars = '***' if abs(coef)/se > 2.58 else ('**' if abs(coef)/se > 1.96 else ('*' if abs(coef)/se > 1.65 else ''))
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{coef:.3f}{stars}', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])
for bar, coef, se in zip(bars2, spec_coefs, spec_ses):
    stars = '***' if abs(coef)/se > 2.58 else ('**' if abs(coef)/se > 1.96 else ('*' if abs(coef)/se > 1.65 else ''))
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{coef:.3f}{stars}', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xticks(x)
ax.set_xticklabels([f'Digital-Intensive\n(N={len(dig):,})', f'Traditional\n(N={len(trad):,})'], fontsize=11)
ax.set_ylabel('Coefficient on ln(Salary)', fontsize=11, color=COLORS['dark'])
ax.set_title('Figure 3: AI Skill Coefficients by Industry Type', fontsize=13, fontweight='bold', color=COLORS['dark'])
ax.legend(fontsize=10, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors=COLORS['dark'])

plt.tight_layout()
save_fig(fig, 'fig_industry_heterogeneity.png')
plt.close()

# ============================================================
# FIGURE 5: Experience Heterogeneity
# ============================================================
print('\n--- Figure 5: Experience Heterogeneity ---')

def run_exp_model(sub):
    sub_y = sub['ln_salary']
    sub_data = pd.DataFrame({
        'AI_general': sub['AI_general'].astype(int),
        'AI_specialized': sub['AI_specialized'].astype(int),
        'Digital': (sub['industry_type'] == 'Digital-Intensive').astype(int),
        'edu_Associate': (sub['education'] == 'Associate').astype(int),
        'edu_Masters': (sub['education'] == 'Masters').astype(int),
        'edu_PhD': (sub['education'] == 'PhD').astype(int),
        'edu_HighSchool': (sub['education'] == 'High School').astype(int),
        'firm_State': (sub['company_type_group'] == 'State-Owned').astype(int),
        'firm_Foreign': (sub['company_type_group'] == 'Foreign').astype(int),
        'firm_Listed': (sub['company_type_group'] == 'Listed').astype(int),
    })
    sub_job = pd.get_dummies(sub['job_category'], prefix='job')
    sub_job = sub_job.drop(columns=[c for c in sub_job.columns if c.endswith('_Other')], errors='ignore')
    sub_X = sm.add_constant(pd.concat([sub_data, sub_job], axis=1).astype(float))
    return sm.OLS(sub_y, sub_X).fit()

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
exp_levels = ['0-1 yr', '1-3 yr', '3+ yr']
exp_labels = ['0–1 Year\n(Entry)', '1–3 Years\n(Mid-Career)', '3+ Years\n(Senior)']

for panel_idx, (skill_var, skill_name, color) in enumerate([
    ('AI_general', 'General-Purpose AI', COLORS['blue']),
    ('AI_specialized', 'Specialized AI', COLORS['crimson'])
]):
    ax = axes[panel_idx]
    coefs, ses, ns = [], [], []
    for exp in exp_levels:
        sub = df[df['experience_level'] == exp]
        m = run_exp_model(sub)
        coefs.append(m.params[skill_var])
        ses.append(m.bse[skill_var])
        ns.append(len(sub))

    ci = [1.96*s for s in ses]
    x = np.arange(3)
    bars = ax.bar(x, coefs, color=color, edgecolor='white', width=0.55,
                  yerr=ci, error_kw={'lw': 1.2, 'capsize': 5, 'capthick': 1.2})

    for bar, coef, se, n in zip(bars, coefs, ses, ns):
        stars = '***' if abs(coef)/se > 2.58 else ('**' if abs(coef)/se > 1.96 else ('*' if abs(coef)/se > 1.65 else ''))
        prem = (np.exp(coef) - 1) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{coef:.3f}{stars}\n(+{prem:.1f}%)', ha='center', fontsize=9,
                fontweight='bold', color=COLORS['dark'])

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.set_ylabel('Coefficient on ln(Salary)', fontsize=11, color=COLORS['dark'])
    ax.set_title(f'({["A","B"][panel_idx]}) {skill_name}', fontsize=13, fontweight='bold', color=COLORS['dark'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=COLORS['dark'])

fig.suptitle('Figure 5: AI Skill Premiums by Experience Level', fontsize=14, fontweight='bold',
             color=COLORS['dark'], y=1.01)
plt.tight_layout()
save_fig(fig, 'fig_experience_heterogeneity.png')
plt.close()

print('\n=== ALL FIGURES REGENERATED ===')
print('Updated: fig_ai_categories.png, fig_baseline_coefficients.png,')
print('         fig_industry_heterogeneity.png, fig_experience_heterogeneity.png')
