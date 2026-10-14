import logging
import re
from datetime import datetime


class Orchestrator:
    def __init__(self, vectordb_client, generation_client, mongo_client,
                 template_parser, embedding_client, reranker_client):
        self.logger            = logging.getLogger(__name__)
        self.generation_client = generation_client
        self.mongo_client      = mongo_client

        from controllers.NLPController import NLPController
        self.nlp_controller = NLPController(
            vectordb_client=vectordb_client,
            generation_client=generation_client,
            embedding_client=embedding_client,
            template_parser=template_parser,
            mongo_client=mongo_client,
            reranker_client=reranker_client
        )

        from agents.VectorDBAgent import VectorDBAgent
        from agents.JobAgent import JobAgent
        from agents.CourseAgent import CourseAgent

        self.vector_agent = VectorDBAgent(
            vectordb_client, generation_client, mongo_client,
            template_parser, embedding_client, reranker_client
        )
        self.job_agent = JobAgent(
            vectordb_client, generation_client, mongo_client,
            template_parser, embedding_client
        )
        self.course_agent = CourseAgent(
            vectordb_client, generation_client, mongo_client,
            template_parser, embedding_client
        )

    # ✅ English to Arabic course name translation
    _ENGLISH_TO_ARABIC = {
    # Core CS courses
    'deep learning':                'التعلم العميق',
    'advanced deep learning':       'التعلم العميق المتقدم',
    'machine learning':             'تعلم الآلة',
    'embedded systems':             'الأنظمة المتضمنة',
    'computer vision':              'معالجة صور رقمية',
    'digital image processing':     'معالجة صور رقمية',
    'neural networks':              'شبكات عصبونية',
    'artificial intelligence':      'مقدمة للذكاء الاصطناعي',
    'intro to ai':                  'مقدمة للذكاء الاصطناعي',
    'data structures':              'خوارزميات وهياكل بيانات',
    'algorithms':                   'خوارزميات وهياكل بيانات',
    'computer networks':            'شبكات الاتصالات',
    'communications networks':      'شبكات الاتصالات',
    'digital design':               'تصميم رقمي',
    'digital logic':                'تصميم رقمي',
    'signal processing':            'معالجة وتحليل الإشارات',
    'signals and systems':          'إشارات ونظم',
    'database':                     'أنظمة قواعد البيانات',
    'database systems':             'أنظمة قواعد البيانات',
    'data management':              'إدارة البيانات',
    'big data':                     'علم البيانات الكبيرة',
    'data science':                 'علم البيانات الكبيرة',
    'robotics':                     'روبوتكس',
    'control':                      'تحكم آلي',
    'automatic control':            'تحكم آلي',
    'electronics':                  'إلكترونيات',
    'circuits':                     'دوائر كهربية',
    'electric circuits':            'دوائر كهربية',
    'statistics':                   'الإحصاء وتحليل البيانات',
    'linear algebra':               'الجبر الخطي',
    'programming':                  'برمجة',
    'project management':           'إدارة المشروعات',
    'discrete math':                'الرياضيات المتقطعة',
    'discrete mathematics':         'الرياضيات المتقطعة',
    'sensors':                      'مستشعرات ومؤثرات',
    'actuators':                    'مستشعرات ومؤثرات',
    'measurements':                 'قياسات وأجهزة قياس',
    'computer architecture':        'معمار الحاسب',
    'computer systems':             'مقدمة لنظم الحاسب',
    'intro to computer':            'مقدمة لنظم الحاسب',
    'power':                        'قوى وآلات كهربية',
    'electric machines':            'قوى وآلات كهربية',
    'technical writing':            'كتابة تقارير فنية',
    'report writing':               'كتابة تقارير فنية',
    'engineering drawing':          'رسم هندسي',
    'drawing':                      'رسم هندسي',
    'chemistry':                    'كيمياء هندسية',
    'engineering chemistry':        'كيمياء هندسية',
    'physics':                      'فيزياء',
    'mechanics':                    'ميكانيكا',
    'math':                         'رياضيات',
    'mathematics':                  'رياضيات',
    'calculus':                     'رياضيات',
    'advanced mathematics':         'رياضيات هندسية متقدمة',
    'mathematical methods':         'طرق رياضية للمهندسين',
    'english':                      'لغة إنجليزية',
    'cognitive psychology':         'علم النفس المعرفي',
    'psychology':                   'علم النفس المعرفي',
    'production engineering':       'هندسة الإنتاج',
    'marketing':                    'التسويق',
    'law':                          'القانون وحقوق الإنسان',
    'human rights':                 'القانون وحقوق الإنسان',
    'ethics':                       'آداب وأخلاقيات المهنة',
    'communication skills':         'مهارات الاتصال والعرض',
    'research skills':              'مهارات البحث والتحليل',
    'history of engineering':       'تاريخ الهندسة والتكنولوجيا',
    'technology history':           'تاريخ الهندسة والتكنولوجيا',
    'ai applications':              'تطبيقات في الذكاء الاصطناعي',
    'practical training':           'تدريب عملي',
    'training':                     'تدريب',
    'graduation project':           'مشروع',
    'project':                      'مشروع',
}

    def _translate_english_terms(self, query: str) -> str:
        query_lower = query.lower()

        # ✅ Handle numbered courses like "physics 1", "math 2", "programming 1"
        numbered_map = {
            'physics':     'فيزياء',
            'mechanics':   'ميكانيكا',
            'math':        'رياضيات',
            'mathematics': 'رياضيات',
            'calculus':    'رياضيات',
            'programming': 'برمجة',
            'english':     'لغة إنجليزية',
            'training':    'تدريب',
            'project':     'مشروع',
            'elective':    'مقرر اختياري',
        }

        for en, ar in numbered_map.items():
            # Match "physics 1", "physics 2", "physics (1)" etc.
            match = re.search(rf'{en}\s*[\(\[]?\s*(\d+)\s*[\)\]]?', query_lower, re.IGNORECASE)
            if match:
                number   = match.group(1)
                original = match.group(0)
                translated = f'{ar} ({number})'
                query = re.sub(re.escape(original), translated, query, flags=re.IGNORECASE)
                print(f"[DEBUG] 🌐 Translated numbered: '{original}' → '{translated}'")

        # Then handle non-numbered terms
        query_lower = query.lower()
        for en, ar in self._ENGLISH_TO_ARABIC.items():
            if en in query_lower:
                query = re.sub(en, ar, query, flags=re.IGNORECASE)
                print(f"[DEBUG] 🌐 Translated: '{en}' → '{ar}'")

        return query

    async def _get_chat_history(self, project_id: str, session_id: str = None):
        try:
            query_filter = {"project_id": project_id}
            if session_id:
                query_filter["session_id"] = session_id

            history_data = await self.mongo_client.db["chat_history"].find(
                query_filter
            ).sort("timestamp", -1).limit(2).to_list(length=2)

            chat_history = []
            for msg in reversed(history_data):
                chat_history.append({"role": "user", "content": msg['query']})
                answer = msg.get('answer', '')[:80].split('،')[0].split('.')[0].strip()
                chat_history.append({"role": "assistant", "content": answer})

            return chat_history
        except Exception as e:
            self.logger.error(f"Error fetching history: {e}")
            return []

    async def route_query_stream(
        self,
        project,
        query: str,
        session_id: str = None,
        limit: int = 5,
        voice_mode: bool = False,
        raw_mode: bool = False
    ):
        if raw_mode:
            refined_query     = query
            effective_history = []
            intent            = "NEW_TOPIC"
            self.logger.info(f"[raw_mode] '{query}'")
        else:
            chat_history = await self._get_chat_history(project.project_id, session_id)
            print(f"\n[DEBUG] 💾 Loaded {len(chat_history)//2} previous turns from MongoDB.")

            analysis      = await self.nlp_controller.rewrite_query(query, chat_history)
            intent        = analysis.get("intent", "NEW_TOPIC")
            refined_query = analysis.get("refined_query", query)
            print(f"[DEBUG] 🪄 NLP Analysis Result: intent={intent}, refined='{refined_query}'")

            if intent == "NEW_TOPIC":
                refined_query     = query
                effective_history = []
                self.logger.info("🧹 Topic Drift Detected — using original query.")
            else:
                effective_history = chat_history[-2:]

        # ✅ Translate English course names to Arabic
        refined_query = self._translate_english_terms(refined_query)

        classification_prompt = f"""
أنت نظام توجيه (Router) آلي صارم جداً. مهمتك الوحيدة هي تصنيف سؤال الطالب إلى فئة واحدة فقط.
يمنع كتابة أي مبررات. الإجابة كلمة واحدة فقط: (ACADEMIC, JOB, COURSE).

دليل التصنيف:
1. ACADEMIC: لأي شيء يخص "كلية الهندسة" أو "جامعة المنصورة" (مواد دراسية، لوائح، ساعات معتمدة، GPA، جداول).
   ⚠️ أي سؤال فيه كلمة "مقرر" أو "مادة" هو ACADEMIC فوراً.
   ⚠️ أي سؤال عن اسم مقرر بالعربي أو الإنجليزي بدون ذكر منصة تعليمية = ACADEMIC فوراً.
   ⚠️ أمثلة ACADEMIC: "deep learning", "machine learning", "embedded systems", "تعلم الاله", "انظمه متضمنه".
2. JOB: للأسئلة عن المسار المهني (سوق العمل، وظائف، تدريبات صيفية، Internships، CV).
3. COURSE: للتعلم الذاتي الخارجي فقط (كورسات أونلاين على Coursera أو Udemy أو YouTube). 
فقط إذا ذكر الطالب صراحةً كلمة "كورس" أو "coursera" أو "udemy" أو "يوتيوب".

السؤال: {refined_query}
التصنيف:"""

        category_res = self.generation_client.generate_response(classification_prompt)
        category     = category_res.strip().upper()
        self.logger.info(f"🚦 Route: {category} | Intent: {intent}")

        if   "JOB"    in category: target_agent = self.job_agent
        elif "COURSE" in category: target_agent = self.course_agent
        else:                      target_agent = self.vector_agent

        full_response = ""

        if hasattr(target_agent, 'process_stream'):
            kwargs = {"chat_history": effective_history, "limit": limit}
            if target_agent is self.vector_agent:
                kwargs["skip_evaluation"] = voice_mode

            async for chunk in target_agent.process_stream(
                project, refined_query, **kwargs
            ):
                full_response += chunk
                yield chunk
        else:
            res           = await target_agent.process(
                project, refined_query,
                chat_history=effective_history, limit=limit
            )
            full_response = res[0]
            yield full_response

        if full_response.strip() and "JOB" not in category and not raw_mode:
            try:
                await self.mongo_client.db["chat_history"].insert_one({
                    "project_id": project.project_id,
                    "session_id": session_id,
                    "query":      query,
                    "answer":     full_response.strip()[:200],
                    "timestamp":  datetime.now()
                })
                self.logger.info("💾 Turn saved to MongoDB.")
            except Exception as e:
                self.logger.error(f"❌ MongoDB save error: {e}")
        elif "JOB" in category:
            self.logger.info("🚫 Skipped JOB turn.")
        elif raw_mode:
            self.logger.info("🚫 Skipped voice tool-call turn.")