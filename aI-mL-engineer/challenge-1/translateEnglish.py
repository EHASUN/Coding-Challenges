import dl_translate as dlt

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."
article_bn = "দ্রুত প্রয়োজন"

mt = dlt.TranslationModel()
translated_fr = mt.translate(article_hi, source=dlt.lang.HINDI, target=dlt.lang.FRENCH)
translated_en = mt.translate(article_ar, source=dlt.lang.ARABIC, target=dlt.lang.ENGLISH)
translated_bn = mt.translate(article_bn, source=dlt.lang.BENGALI, target=dlt.lang.ENGLISH)
print(translated_fr)