from datetime import date
from io import BytesIO
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO

st.set_page_config(page_title='Өкпе кескіндерін талдау | ЖИ агенті', page_icon='🫁', layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1250px}.stButton>button[kind="primary"]{background:#087e8b;border-color:#087e8b}h1,h2,h3{letter-spacing:-.02em}</style>''', unsafe_allow_html=True)
st.title('🫁 Өкпе кескіндерін талдауға арналған ЖИ агенті')
st.caption('YOLO негізіндегі зерттеу прототипі • Интерфейс тілі: қазақша')
st.warning('Бұл жүйе зерттеу және демонстрация мақсаттарына арналған. Нәтиже клиникалық диагноз болып саналмайды. Туберкулезді растау немесе жоққа шығару үшін дәрігердің бағалауы және тиісті зерттеулер қажет.')

@st.cache_resource
def load_model():
    path = Path(__file__).parent / 'weights' / 'best.pt'
    return YOLO(str(path))

try:
    model = load_model()
except Exception as exc:
    st.error('Модельді жүктеу мүмкін болмады. weights/best.pt файлын және кітапхана нұсқаларын тексеріңіз.')
    st.exception(exc)
    st.stop()

names = model.names
if isinstance(names, list):
    names = dict(enumerate(names))

with st.sidebar:
    st.header('Навигация')
    section = st.radio('Бөлімді таңдаңыз', ['Пациент және талдау', 'Модель туралы', 'Қолдану нұсқаулығы'], label_visibility='collapsed')
    st.divider()
    st.header('Талдау параметрлері')
    conf = st.slider('Сенімділік шегі', .05, .95, .25, .05, help='Осы мәннен төмен болжамдар көрсетілмейді.')
    iou = st.slider('Қабаттасу шегі (IoU)', .10, .95, .50, .05)
    show_boxes = st.checkbox('Анықталған аймақтарды көрсету', True)
    st.caption('Бұл шектер модельдің клиникалық сезімталдығы немесе ерекшелігі емес.')

if section == 'Модель туралы':
    st.subheader('Модель туралы мәлімет')
    st.write('**Архитектура:** YOLO (жүктелген best.pt салмақтары)')
    st.write('**Модельдегі сыныптар:** ' + ', '.join(f'{k}: {v}' for k,v in names.items()))
    st.info('Дәлдік (precision), толықтық (recall), mAP50 және mAP50–95 көрсеткіштері бұл файлдардан расталмаған. Оларды тәуелсіз белгіленген тексеру жиынында есептеу қажет. Жеке кескіннің сенімділік көрсеткіші диагностикалық дәлдікке тең емес.')
    st.write('**Шектеу:** бұл нұсқа PNG/JPG форматындағы бір кескінді өңдейді; толық КТ томографиясын немесе DICOM сериясын талдамайды. КТ-ға қолдану тек модельдің КТ деректерінде үйретіліп, тексерілгені расталғаннан кейін орынды.')
    st.stop()
if section == 'Қолдану нұсқаулығы':
    st.subheader('Қолдану тәртібі')
    st.markdown('1. Пациенттің деректерін енгізіңіз.\n2. Модель оқытылған кескін түріне сәйкес PNG/JPG суретін жүктеңіз.\n3. «Кескінді талдау» түймесін басыңыз.\n4. Нәтижені қарап, қажет болса есепті жүктеңіз.\n5. Қорытындыны дәрігерге көрсетіңіз.')
    st.info('Пациент деректері бұл нұсқада дерекқорға сақталмайды. Нақты пациент деректерін енгізгенде құпиялылық және қолжетімділік талаптарын сақтаңыз.')
    st.stop()

st.warning('Демонстрациялық нұсқа. Тек ойдан шығарылған пациент деректерін және дербес мәліметтері жойылған кескіндерді пайдаланыңыз.')

st.subheader('1. Пациент туралы мәліметтер')
with st.form('patient_form'):
    a,b,c = st.columns(3)
    with a:
        fio = st.text_input('Пациенттің аты-жөні *', placeholder='Тегі Аты Әкесінің аты')
        birth = st.date_input('Туған күні', value=None, min_value=date(1900,1,1), max_value=date.today(), format='DD.MM.YYYY')
    with b:
        sex = st.selectbox('Жынысы', ['Көрсетілмеген', 'Әйел', 'Ер'])
        study_date = st.date_input('Зерттеу күні *', value=date.today(), max_value=date.today(), format='DD.MM.YYYY')
    with c:
        record = st.text_input('Зерттеу коды (міндетті емес)', placeholder='Мысалы: TB-001')
        modality = st.selectbox('Кескін түрі', ['Рентгенография (CXR)', 'КТ кескінінің жеке қимасы'])
    st.subheader('2. Медициналық кескінді жүктеу')
    upload = st.file_uploader('Өкпе кескінін таңдаңыз (PNG, JPG, JPEG)', type=['png','jpg','jpeg'])
    submitted = st.form_submit_button('🔎 Кескінді талдау', type='primary', use_container_width=True)

if submitted:
    if not fio.strip():
        st.error('Пациенттің аты-жөнін енгізіңіз.')
        st.stop()
    if upload is None:
        st.error('Талдау үшін кескінді жүктеңіз.')
        st.stop()
    if modality.startswith('КТ'):
        st.error('КТ кескінін талдау әзірше қосылмаған: жүктелген модельдің КТ деректерінде оқытылғаны және тексерілгені расталуы қажет.')
        st.stop()
    try:
        img = ImageOps.exif_transpose(Image.open(upload)).convert('RGB')
        if img.width < 64 or img.height < 64:
            st.error('Кескін өлшемі тым кіші.'); st.stop()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error('Кескінді оқу мүмкін болмады. Дұрыс PNG/JPG файлын таңдаңыз.')
        st.stop()
    with st.spinner('ЖИ моделі кескінді талдап жатыр…'):
        try:
            res = model.predict(source=np.asarray(img), conf=conf, iou=iou, verbose=False)[0]
        except Exception as exc:
            st.error('Талдау кезінде қате пайда болды.'); st.exception(exc); st.stop()
    # Only explicit TB-related class names count as a TB finding.
    tb_terms = ('tb','tuberculosis','tuberkulez','туберкулез','туберкулёз')
    tb_ids = [int(k) for k,v in names.items() if str(v).strip().lower() in tb_terms]
    detections = []
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls.item())
            detections.append((cls_id, float(box.conf.item())))
    positives = [score for cls_id,score in detections if cls_id in tb_ids]
    st.divider()
    st.subheader('3. Талдау нәтижесі')
    if not tb_ids:
        st.error('Модель сыныптарында туберкулезге сәйкес атау табылмады. Нәтижені түсіндіру тоқтатылды. Модельдің сынып атауларын тексеріңіз.')
        st.write('Модель сыныптары:', names)
    elif positives:
        st.error('⚠️ Туберкулезге күмәнді белгілер анықталды')
        st.warning('Ұсыным: жақын арада фтизиатрға немесе дәрігерге қаралыңыз. Қажетті қосымша зерттеулерді дәрігер анықтайды. Бұл нәтиже туберкулез диагнозын растамайды.')
    else:
        st.info('ℹ️ Берілген кескінде модель туберкулезге күмәнді белгілерді анықтамады')
        st.warning('Ұсыным: теріс нәтиже туберкулезді жоққа шығармайды. Симптомдар немесе қауіп факторлары болса, дәрігерге қаралыңыз.')
    x,y,z = st.columns(3)
    x.metric('Анықталған күмәнді аймақтар', len(positives) if tb_ids else '—')
    y.metric('Ең жоғары сенімділік', f'{max(positives)*100:.1f}%' if positives else '—')
    z.metric('Талдау шегі', f'{conf*100:.0f}%')
    left,right = st.columns(2)
    with left:
        st.image(img, caption='Бастапқы кескін', use_column_width=True)
    with right:
        if show_boxes:
            plotted = res.plot()
            st.image(plotted, channels='BGR', caption='Модель белгілеген аймақтар', use_column_width=True)
        else:
            st.info('Аймақтарды көрсету өшірілген.')
    st.caption('Сенімділік пайызы — жеке болжамның бағасы, диагноздың ықтималдығы немесе модельдің жалпы дәлдігі емес.')
    if tb_ids:
        status = 'Туберкулезге күмәнді белгілер анықталды' if positives else 'Күмәнді белгілер анықталмады (ауру жоққа шығарылмайды)'
        report = '\n'.join(['ЖИ ТАЛДАУЫНЫҢ ЗЕРТТЕУЛІК ЕСЕБІ',f'Пациент: {fio.strip()}',f'Туған күні: {birth.strftime("%d.%m.%Y") if birth else "Көрсетілмеген"}',f'Жынысы: {sex}',f'Зерттеу күні: {study_date.strftime("%d.%m.%Y")}',f'Зерттеу коды: {record or "—"}',f'Кескін түрі: {modality}',f'Нәтиже: {status}',f'Күмәнді аймақ саны: {len(positives)}',f'Ең жоғары сенімділік: {max(positives)*100:.1f}%' if positives else 'Ең жоғары сенімділік: —','Ұсыным: дәрігерлік бағалау және қажет болған жағдайда қосымша тексеру.','Ескерту: бұл клиникалық диагноз емес.'])
        st.download_button('📄 Талдау есебін жүктеу (TXT)', data=report.encode('utf-8-sig'), file_name='talдау_esebi.txt', mime='text/plain')
