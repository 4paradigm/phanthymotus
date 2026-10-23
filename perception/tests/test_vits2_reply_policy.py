"""Pure language policy regression, independent of ROS/model synthesis."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import threading
import unittest


class LanguagePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from plugins.vits2_tts_language import policy
        cls.module = policy

    def setUp(self):
        self.policy = self.module.LanguagePolicy()

    def assert_code(self, code, operation):
        with self.assertRaises(self.module.LanguagePolicyError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)

    def test_auto_follows_current_input_and_switches_back(self):
        tokens = []
        for text, expected, answer in (
            ("hello", "en", "Hello, how can I help you?"),
            ("你好", "zh", "你好，请问有什么可以帮助你的？"),
            ("hello again", "en", "Welcome back."),
        ):
            result = self.policy.begin_turn(text)
            self.assertEqual(result["reply_language"], expected)
            self.assertEqual(result["detected_language"], expected)
            request = self.policy.prepare(answer, result["turn_id"])
            self.assertEqual((request.text, request.language), (answer, expected))
            tokens.append(result["turn_id"])
        self.assertEqual(len(set(tokens)), 3)

    def test_fixed_modes_override_input_language(self):
        for mode, text, answer in (("zh", "hello", "你好。"), ("en", "你好", "Hello.")):
            with self.subTest(mode=mode):
                policy = self.module.LanguagePolicy(mode)
                result = policy.begin_turn(text)
                self.assertEqual(result["reply_language"], mode)
                self.assertEqual(policy.prepare(answer, result["turn_id"]).language, mode)

    def test_instruction_is_available_before_generating_reply(self):
        for text, language_name in (("hello", "English"), ("你好", "Chinese")):
            result = self.policy.begin_turn(text)
            self.assertIn("mainly in " + language_name, result["instruction"])
            self.assertIn("mixing is allowed", result["instruction"])
            self.assertNotIn("entirely", result["instruction"])
            self.assertIsInstance(result["turn_id"], str)

    def test_speak_cannot_infer_language_from_generated_reply(self):
        self.assert_code("language_turn_required", lambda: self.policy.prepare("Hello", "abc"))

    def test_old_token_is_rejected_after_switch(self):
        first = self.policy.begin_turn("hello")
        second = self.policy.begin_turn("你好")
        self.assert_code("stale_turn", lambda: self.policy.prepare("Hello", first["turn_id"]))
        self.assertEqual(self.policy.prepare("你好", second["turn_id"]).language, "zh")

    def test_new_policy_cannot_reuse_old_policy_token(self):
        old = self.policy.begin_turn("hello")
        replacement = self.module.LanguagePolicy()
        current = replacement.begin_turn("hello")
        self.assertNotEqual(old["turn_id"], current["turn_id"])
        self.assert_code("stale_turn", lambda: replacement.prepare("Hello", old["turn_id"]))

    def test_config_change_invalidates_pending_reply(self):
        first = self.policy.begin_turn("hello")
        self.policy.configure("zh")
        self.assertIsNone(self.policy.snapshot()["remembered_language"])
        self.assert_code("language_turn_required", lambda: self.policy.prepare("Hello", first["turn_id"]))
        next_turn = self.policy.begin_turn("hello")
        self.assertEqual(next_turn["reply_language"], "zh")

    def test_same_mode_config_keeps_valid_turn(self):
        first = self.policy.begin_turn("hello")
        self.policy.configure("auto")
        self.assertEqual(self.policy.prepare("Hello", first["turn_id"]).language, "en")
        next_turn = self.policy.begin_turn("2026 10/23")
        self.assertEqual((next_turn["reply_language"], next_turn["language_source"]), ("en", "history"))

    def test_invalid_input_and_config_do_not_mutate_accepted_turn(self):
        accepted = self.policy.begin_turn("hello")
        before = self.policy.snapshot()
        operations = (
            ("empty_text", lambda: self.policy.begin_turn(" ")),
            ("empty_text", lambda: self.policy.begin_turn(None)),
            ("invalid_input_language", lambda: self.policy.begin_turn("hello", "fr")),
            ("invalid_requested_reply_language", lambda: self.policy.begin_turn(
                "hello", requested_reply_language="auto")),
            ("unsupported_script", lambda: self.policy.begin_turn("สวัสดี")),
            ("invalid_reply_language", lambda: self.policy.configure("fr")),
        )
        for code, operation in operations:
            with self.subTest(code=code):
                self.assert_code(code, operation)
                self.assertEqual(self.policy.snapshot(), before)
        self.assertEqual(self.policy.prepare("Hello", accepted["turn_id"]).language, "en")

    def test_explicit_asr_language_resolves_mixed_first_input(self):
        for text in ("你好 hello", "12345"):
            for language in ("zh", "en"):
                with self.subTest(text=text, language=language):
                    policy = self.module.LanguagePolicy()
                    result = policy.begin_turn(text, input_language=language)
                    self.assertEqual(result["reply_language"], language)
                    self.assertEqual(result["detected_language"], language)
                    self.assertEqual(result["language_source"], "asr")

    def test_auto_uses_english_grammar_around_chinese_names(self):
        for text in (
            "Please check 北京 and tell me the result.",
            "We saw Qwen at 北京.",
            "Can you explain 北京交通大学人工智能与机器人研究中心 to me?",
            "What is 北京?",
            "Please explain 北京.",
            "Tell me about 北京.",
            "Please explain 北京交通大学人工智能与机器人研究中心.",
            "北京 is beautiful.",
            "北京 has a large network and is ready.",
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual((turn["reply_language"], turn["detected_language"]), ("en", "en"))
                self.assertEqual(turn["language_source"], "text")

    def test_auto_keeps_chinese_grammar_with_many_english_product_names(self):
        for text in (
            "我们要检查 OpenMLDB Qwen SQL Playground 和 FeatInsight。",
            "请使用 NVIDIA TensorRT OpenMLDB FeatInsight Qwen 进行测试。",
            "我们在waic会议上看到许多产品和模型，qwen、豆包等都有出现",
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], "zh")
                self.assertEqual(turn["language_source"], "text")

    def test_ambiguous_names_and_unquoted_titles_do_not_force_english(self):
        for text in ("你好 hello", "北京 server 8.5GB", "请看 The Lord of the Rings。",
                     "我想读 The Lord of the Rings。", "介绍 The Lord of the Rings。",
                     "推荐 The Lord of the Rings。", "讲讲 The Lord of the Rings。",
                     "聊聊 The Lord of the Rings。", "谈谈 The Lord of the Rings。",
                     "读一读 The Lord of the Rings。"):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], "zh")
                self.assertIsNone(turn["detected_language"])
                self.assertEqual(turn["language_source"], "default")

    def test_conflicting_bilingual_clauses_do_not_inherit_prior_english(self):
        for text in ("Please check the network. 然后我们检查电量。",
                     "We will check the network, then report the result. 请检查电量。"):
            with self.subTest(text=text):
                previous = self.policy.begin_turn("hello")
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], "zh")
                self.assertNotEqual(turn["turn_id"], previous["turn_id"])

    def test_quotes_and_parentheses_do_not_override_outer_language(self):
        for text, expected in (
            ('请读英文“Please check the network and tell me the result.”', "zh"),
            ('Please read “请检查网络然后告诉我们结果。” for me.', "en"),
            ('Please read (中文“请检查网络”) for me.', "en"),
            ('请念英文“Please check the network', "zh"),
            ('《The Lord of the Rings》是什么？', "zh"),
            ('介绍《The Lord of the Rings》。', "zh"),
            ("It's 北京, and the users' network isn't ready.", "en"),
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], expected)
                self.assertEqual(turn["language_source"], "text")

    def test_addresses_and_numeric_entities_are_not_language_evidence(self):
        for text, expected, source in (
            ("请查看 https://we.are.at.example.com/this/is/our/path 和 me@our.example.com。", "zh", "text"),
            ("Please check 北京 at https://example.com/qwen3 and tell me the result.", "en", "text"),
            ("https://we.are.at.example.com/this/is/our/path", "en", "history"),
            ("2026-08-07 14:25 RTX-4090 8.5GB", "en", "history"),
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual((turn["reply_language"], turn["language_source"]), (expected, source))

    def test_asr_label_overrides_conflicting_text_grammar(self):
        for text, language in (
            ("Please check 北京 and tell me the result.", "zh"),
            ("请检查 OpenMLDB 和 Qwen。", "en"),
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text, input_language=language)
                self.assertEqual((turn["reply_language"], turn["language_source"]), (language, "asr"))

    def test_current_user_requested_language_overrides_asr_and_fixed_preference(self):
        for mode in ("auto", "zh", "en"):
            for language, opposite in (("zh", "en"), ("en", "zh")):
                with self.subTest(mode=mode, language=language):
                    policy = self.module.LanguagePolicy(mode)
                    turn = policy.begin_turn("hello", input_language=opposite,
                                             requested_reply_language=language)
                    self.assertEqual((turn["reply_language"], turn["language_source"]), (language, "requested"))
                    self.assertEqual(turn["detected_language"], opposite)
                    self.assertEqual(policy.snapshot()["reply_language"], mode)

    def test_requested_language_is_one_turn_only_and_never_changes_configuration(self):
        for mode, following in (("auto", "zh"), ("zh", "zh"), ("en", "en")):
            with self.subTest(mode=mode):
                policy = self.module.LanguagePolicy(mode)
                first = policy.begin_turn("你好", requested_reply_language="en")
                request = policy.prepare("Hello.", first["turn_id"])
                second = policy.begin_turn("你好")
                self.assertEqual(second["reply_language"], following)
                self.assertEqual(policy.snapshot()["reply_language"], mode)
                self.assertEqual((request.text, request.language), ("Hello.", "en"))
                self.assertNotEqual(first["turn_id"], second["turn_id"])

    def test_arbitrary_language_instructions_require_caller_interpretation(self):
        for text, detected, requested in (
            ("Please answer in Chinese.", "en", "zh"),
            ("请用英文回答。", "zh", "en"),
        ):
            with self.subTest(text=text):
                self.assertEqual(self.policy.begin_turn(text)["reply_language"], detected)
                turn = self.policy.begin_turn(text, requested_reply_language=requested)
                self.assertEqual((turn["reply_language"], turn["language_source"]), (requested, "requested"))

    def test_first_mixed_or_numeric_auto_input_defaults_to_chinese(self):
        for text in ("你好 hello", "12345", "🙂!?"):
            with self.subTest(text=text):
                policy = self.module.LanguagePolicy()
                result = policy.begin_turn(text)
                self.assertEqual(result["reply_language"], "zh")
                self.assertIsNone(result["detected_language"])
                self.assertEqual(result["language_source"], "default")
                self.assertIn("Chinese", result["instruction"])
                self.assertEqual(policy.prepare("你好。", result["turn_id"]).language, "zh")

    def test_neutral_input_inherits_english_but_ambiguous_prose_keeps_fallback(self):
        for text, language, source in (("12345", "en", "history"),
                                      ("2026 10/23", "en", "history"),
                                      ("你好 hello", "zh", "default"),
                                      ("🙂!?", "zh", "default")):
            with self.subTest(text=text):
                previous = self.policy.begin_turn("hello")
                result = self.policy.begin_turn(text)
                self.assertEqual((result["reply_language"], result["language_source"]), (language, source))
                self.assertIsNone(result["detected_language"])
                reply = "Hello." if language == "en" else "你好。"
                self.assertEqual(self.policy.prepare(reply, result["turn_id"]).language, language)
                self.assert_code(
                    "stale_turn", lambda: self.policy.prepare("Hello", previous["turn_id"]),
                )

    def test_auto_remembers_latest_clear_language_across_neutral_turns(self):
        for text, language, source, detected in (
            ("2026 10/23", "zh", "default", None),
            ("Hello", "en", "text", "en"),
            ("2026 10/23", "en", "history", None),
            ("y=2x-3", "en", "history", None),
            ("user@example.com", "en", "history", None),
            ("你好", "zh", "text", "zh"),
            ("RTX-4090", "zh", "history", None),
            ("2/3", "zh", "history", None),
        ):
            with self.subTest(text=text):
                turn = self.policy.begin_turn(text)
                self.assertEqual((turn["reply_language"], turn["language_source"], turn["detected_language"]),
                                 (language, source, detected))

    def test_auto_asr_and_explicit_selection_update_neutral_followup_language(self):
        for arguments, source in (({"input_language": "en"}, "asr"),
                                  ({"requested_reply_language": "en"}, "requested")):
            with self.subTest(arguments=arguments):
                policy = self.module.LanguagePolicy()
                # Direct speech's all-Latin fallback also uses an explicit
                # requested language; it must not be lost on the next number.
                first = policy.begin_turn("A100", **arguments)
                self.assertEqual((first["reply_language"], first["language_source"]), ("en", source))
                following = policy.begin_turn("2026 10/23")
                self.assertEqual((following["reply_language"], following["language_source"]), ("en", "history"))
                self.assertIsNone(following["detected_language"])

    def test_changed_mode_and_recreated_policy_start_without_memory(self):
        self.policy.begin_turn("Hello")
        self.policy.configure("zh")
        self.policy.begin_turn("Hello")
        self.assertIsNone(self.policy.snapshot()["remembered_language"])
        self.policy.configure("auto")
        turn = self.policy.begin_turn("2026 10/23")
        self.assertEqual((turn["reply_language"], turn["language_source"]), ("zh", "default"))
        self.policy.begin_turn("Hello")
        replacement = self.module.LanguagePolicy()
        fresh = replacement.begin_turn("2026 10/23")
        self.assertEqual((fresh["reply_language"], fresh["language_source"]), ("zh", "default"))

    def test_fixed_preferences_do_not_record_inputs_or_one_turn_overrides(self):
        for mode, other in (("zh", "en"), ("en", "zh")):
            with self.subTest(mode=mode):
                policy = self.module.LanguagePolicy(mode)
                fixed = policy.begin_turn("2026 10/23", input_language=other)
                self.assertEqual((fixed["reply_language"], fixed["language_source"]), (mode, "fixed"))
                policy.begin_turn("2026 10/23", requested_reply_language=other)
                self.assertIsNone(policy.snapshot()["remembered_language"])
                policy.configure("auto")
                fresh = policy.begin_turn("2026 10/23")
                self.assertEqual((fresh["reply_language"], fresh["language_source"]), ("zh", "default"))

    def test_ambiguous_prose_and_generated_replies_do_not_replace_language_memory(self):
        accepted = self.policy.begin_turn("Hello")
        before = self.policy.snapshot()
        self.policy.prepare("Hello, 我们检查机器人的状态。", accepted["turn_id"])
        self.assertEqual(self.policy.snapshot(), before)
        ambiguous = self.policy.begin_turn("你好 hello")
        self.assertEqual((ambiguous["reply_language"], ambiguous["language_source"]), ("zh", "default"))
        following = self.policy.begin_turn("2026 10/23")
        self.assertEqual((following["reply_language"], following["language_source"]), ("en", "history"))

    def test_fixed_mode_accepts_first_neutral_input(self):
        for mode in ("zh", "en"):
            for text in ("12345", "你好 hello"):
                with self.subTest(mode=mode, text=text):
                    policy = self.module.LanguagePolicy(mode)
                    result = policy.begin_turn(text)
                    self.assertEqual(result["reply_language"], mode)

    def test_wrong_reply_script_rejected_without_text_rewriting(self):
        for user, bad_replies in (
            ("hello", ("你好", "Привет", "สวัสดี", "Hello 你好 สวัสดี")),
            ("你好", ("Hello", "こんにちは", "你好 hello Привет")),
        ):
            result = self.policy.begin_turn(user)
            before = self.policy.snapshot()
            for reply in bad_replies:
                with self.subTest(reply=reply):
                    self.assert_code(
                        "reply_language_mismatch",
                        lambda: self.policy.prepare(reply, result["turn_id"]),
                    )
                    self.assertEqual(self.policy.snapshot(), before)

    def test_mixed_reply_preserves_text_for_either_primary_language(self):
        for user, reply, language in (
            ("你好", "我们在waic会议上看到许多产品和模型，qwen、豆包等都有出现", "zh"),
            ("你好", "你好，please wait here. 我马上回来。", "zh"),
            ("hello", "Hello，欢迎来到北京。Please wait here.", "en"),
        ):
            with self.subTest(user=user, reply=reply):
                turn = self.policy.begin_turn(user)
                request = self.policy.prepare(reply, turn["turn_id"])
                self.assertEqual((request.text, request.language), (reply, language))
                self.assertEqual(request.turn_id, turn["turn_id"])

    def test_standalone_known_names_keep_original_text_in_both_languages(self):
        for language in ("zh", "en"):
            policy = self.module.LanguagePolicy(language)
            turn = policy.begin_turn("Qwen3")
            for raw in ("Qwen", "qwen", "Qwen3", "QWEN3", "Qwen2.5", " \tQwen3\n",
                        "Qwen 3", "Qwen 2.5", "QWEN\t3", "Qwen\u00a03", "Qwen\u30003。"):
                with self.subTest(language=language, raw=raw):
                    request = policy.prepare(raw, turn["turn_id"])
                    self.assertEqual((request.text, request.language, request.turn_id),
                                     (raw, language, turn["turn_id"]))

    def test_standalone_curated_names_do_not_supply_auto_language_evidence(self):
        for raw in ("Qwen", "Qwen3", "Qwen 3", "Qwen 2.5", "QWEN\t3", "Qwen\u00a03!", "Qwen\u30003。"):
            with self.subTest(raw=raw):
                self.policy.begin_turn("Hello")
                turn = self.policy.begin_turn(raw)
                self.assertEqual((turn["reply_language"], turn["language_source"]), ("en", "history"))
                self.assertIsNone(turn["detected_language"])
                self.assertEqual(self.policy.prepare(raw, turn["turn_id"]).text, raw)
                for supplied in ("zh", "en"):
                    tagged = self.policy.begin_turn(raw, input_language=supplied)
                    self.assertEqual((tagged["reply_language"], tagged["language_source"]), (supplied, "asr"))

    def test_chinese_name_exception_rejects_clauses_and_partial_names(self):
        policy = self.module.LanguagePolicy("zh")
        turn = policy.begin_turn("Qwen3")
        before = policy.snapshot()
        for raw in (
            "myqwen", "qwenish", "Qwen3x", "Qwen3_abc", "Qwen3/ABC",
            "Qwen3 is ready", "This is Qwen3", "Qwen3 Qwen3",
            "Qwen 3x", "Qwen 3_abc", "Qwen 3/ABC", "Qwen 3 is ready",
            "This is Qwen 3", "Qwen 3 Qwen 3", "Qwen\n3",
        ):
            with self.subTest(raw=raw):
                self.assert_code(
                    "reply_language_mismatch",
                    lambda: policy.prepare(raw, turn["turn_id"]),
                )
                self.assertEqual(policy.snapshot(), before)

    def test_english_spaced_versions_preserve_name_boundaries_and_identifiers(self):
        from plugins.vits2_tts_language.name_text import normalize_standalone_name

        for raw, expected in (
            ("Qwen 3", "Qwen three"),
            ("Qwen\t2.5", "Qwen two point five"),
            ("Qwen\u00a03!", "Qwen three!"),
            ("Qwen 3 is ready.", "Qwen three is ready."),
            ("你好，Qwen 3 is ready，然后继续。", "你好，Qwen three is ready，然后继续。"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_standalone_name(raw, "en"), expected)
        for raw in (
            "myqwen 3", "qwenish 3", "Qwen 3x", "Qwen 3_abc", "Qwen 3/ABC",
            "Qwen 3.5.1", "Qwen\n3", "https://qwen.ai/Qwen3", "qwen3@example.com",
            "order id: Qwen3", "order id: Qwen 3", "model_id=Qwen 3",
        ):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_standalone_name(raw, "en"), raw)

    def test_request_is_frozen_and_preserves_original_text(self):
        result = self.policy.begin_turn("hello")
        text = "Hello, the value is 1.2."
        request = self.policy.prepare(text, result["turn_id"])
        self.assertEqual(request.text, text)
        self.policy.begin_turn("你好")
        self.assertEqual(request.language, "en")
        self.assertEqual(request.turn_id, result["turn_id"])
        with self.assertRaises(FrozenInstanceError):
            request.language = "zh"

    def test_neutral_numeric_reply_preserves_raw_text_and_selected_language(self):
        for utterance, language in (("hello", "en"), ("你好", "zh")):
            turn = self.policy.begin_turn(utterance)
            for reply in ("123", "1.2", "3/4", "2026 10/ 20", "－１２℃"):
                with self.subTest(utterance=utterance, reply=reply):
                    request = self.policy.prepare(reply, turn["turn_id"])
                    self.assertEqual((request.text, request.language), (reply, language))
                    self.assertEqual(request.turn_id, turn["turn_id"])

    def test_formula_variables_and_model_prefixes_are_not_english_prose(self):
        from plugins.vits2_tts_language.neutral_text import is_neutral_symbolic

        for text in ("y = 2x‑3", "RTX‑4090", "RTX4090", "1.2A", "SNR≥28dB",
                     "lr=1e‑4", "NO.0917", "x^2", "x=1e-4", "x=2kg"):
            with self.subTest(text=text):
                self.assertTrue(is_neutral_symbolic(text))
                turn = self.policy.begin_turn(text)
                self.assertEqual(turn["reply_language"], "zh")
                self.assertEqual(turn["language_source"], "default")
                self.assertIsNone(turn["detected_language"])
                request = self.policy.prepare(text, turn["turn_id"])
                self.assertEqual(request.text, text)
                self.assertEqual(request.language, "zh")

    def test_symbolic_detection_does_not_reinterpret_prose_names_or_partial_fields(self):
        from plugins.vits2_tts_language.neutral_text import is_neutral_symbolic

        for text in ("The equation is y=2x-3.", "公式 y=2x-3。", "A100", "Qwen 3",
                     "CAR", "AI", "I/O", "2kg/day", "3.05kgModel", "100MBps",
                     "Visit https://example.com/2/3", "请访问https://example.com。",
                     "Email user@example.com", "请联系user@example.com。",
                     "https://", "user@", "x^", "(x+2"):
            with self.subTest(text=text):
                self.assertFalse(is_neutral_symbolic(text))

    def test_complete_addresses_follow_fixed_mode_or_auto_language_memory(self):
        from plugins.vits2_tts_language.neutral_text import is_neutral_symbolic, normalize_neutral_text

        for mode in ("zh", "en", "auto"):
            policy = self.module.LanguagePolicy(mode)
            policy.begin_turn("Hello, the robot is ready.")
            for text, zh_fragment, en_fragment in (
                ("https://example.com", "冒号斜杠斜杠", "colon slash slash"),
                ("user@example.com", "艾特", " at "),
                ("https://example.com/2026/10/20?q=2/3", "二斜杠三", "two slash three"),
                ("user+qwen3@example.com", "Q W E N 三艾特", "Q W E N three at"),
                ("https://qwen.ai/Qwen3", "Q W E N 三", "Q W E N three"),
                ("qwen3@example.com", "Q W E N 三艾特", "Q W E N three at"),
            ):
                with self.subTest(mode=mode, text=text):
                    self.assertTrue(is_neutral_symbolic(text))
                    turn = policy.begin_turn(text)
                    language = "zh" if mode == "zh" else "en"
                    self.assertEqual(turn["reply_language"], language)
                    self.assertEqual(turn["language_source"], "history" if mode == "auto" else "fixed")
                    self.assertIsNone(turn["detected_language"])
                    request = policy.prepare(text, turn["turn_id"])
                    self.assertEqual(request.text, text)
                    normalized = normalize_neutral_text(request.text, request.language)
                    self.assertIn(en_fragment if language == "en" else zh_fragment, normalized)
                    self.assertNotIn("October", normalized)
                    self.assertNotIn("two thirds", normalized)
                    self.assertNotIn("分之", normalized)
                    self.assertNotIn("千问", normalized)

    def test_unseen_symbolic_values_use_the_existing_format_readers(self):
        from plugins.vits2_tts_language.neutral_text import normalize_neutral_text

        for text, zh, en in (
            ("z=4a+7", "z 等于四乘以 a 加七", "z equals four times a plus seven"),
            ("GTX-1650", "G T X 一六五零", "G T X one six five zero"),
            ("0.75mA", "零点七五毫安", "zero point seven five milliamperes"),
            ("5/8", "八分之五", "five eighths"),
            ("x=1e-4", "x 等于一乘以十的负四次方",
             "x equals one times ten to the power of minus four"),
            ("x=2kg", "x 等于二千克", "x equals two kilograms"),
        ):
            with self.subTest(text=text):
                self.assertEqual(normalize_neutral_text(text, "zh"), zh)
                self.assertEqual(normalize_neutral_text(text, "en"), en)

    def test_punctuation_and_emoji_only_reply_still_require_spoken_words(self):
        for utterance in ("hello", "你好"):
            turn = self.policy.begin_turn(utterance)
            for reply in ("?!", "🙂", "－℃", " / "):
                with self.subTest(utterance=utterance, reply=reply):
                    self.assert_code("ambiguous_reply_language", lambda: self.policy.prepare(
                        reply, turn["turn_id"]
                    ))

    def test_sessions_are_independent(self):
        other = self.module.LanguagePolicy()
        a = self.policy.begin_turn("hello")
        b = other.begin_turn("你好")
        self.assertEqual(self.policy.prepare("Hello", a["turn_id"]).language, "en")
        self.assertEqual(other.prepare("你好", b["turn_id"]).language, "zh")
        self.assert_code("stale_turn", lambda: other.prepare("你好", a["turn_id"]))
        self.assertEqual(self.policy.begin_turn("2026 10/23")["reply_language"], "en")
        self.assertEqual(other.begin_turn("2026 10/23")["reply_language"], "zh")

    def test_concurrent_turn_switch_never_relabels_an_accepted_reply(self):
        for _ in range(20):
            policy = self.module.LanguagePolicy()
            first = policy.begin_turn("hello")
            barrier = threading.Barrier(2)

            def prepare():
                barrier.wait(timeout=3)
                try:
                    return policy.prepare("Hello", first["turn_id"])
                except self.module.LanguagePolicyError as exc:
                    return exc.code

            def switch():
                barrier.wait(timeout=3)
                return policy.begin_turn("你好")

            with ThreadPoolExecutor(max_workers=2) as pool:
                pending = pool.submit(prepare)
                switched = pool.submit(switch)
                result = pending.result(timeout=5)
                switched.result(timeout=5)
            if isinstance(result, str):
                self.assertEqual(result, "stale_turn")
            else:
                self.assertEqual((result.text, result.language, result.turn_id),
                                 ("Hello", "en", first["turn_id"]))
            self.assertEqual(policy.snapshot()["resolved_language"], "zh")
