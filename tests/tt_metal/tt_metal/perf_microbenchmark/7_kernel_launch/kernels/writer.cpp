// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t compile_arg0 = get_compile_time_arg_val(0);

    uint32_t runtime_arg0 = get_arg_val<uint32_t>(0);
    uint32_t runtime_arg1 = get_arg_val<uint32_t>(1);
    uint32_t runtime_arg2 = get_arg_val<uint32_t>(2);
    uint32_t runtime_arg3 = get_arg_val<uint32_t>(3);
    uint32_t runtime_arg4 = get_arg_val<uint32_t>(4);
    uint32_t runtime_arg5 = get_arg_val<uint32_t>(5);
    uint32_t runtime_arg6 = get_arg_val<uint32_t>(6);
    uint32_t runtime_arg7 = get_arg_val<uint32_t>(7);
    uint32_t runtime_arg8 = get_arg_val<uint32_t>(8);
    uint32_t runtime_arg9 = get_arg_val<uint32_t>(9);
    uint32_t runtime_arg10 = get_arg_val<uint32_t>(10);
    uint32_t runtime_arg11 = get_arg_val<uint32_t>(11);
    uint32_t runtime_arg12 = get_arg_val<uint32_t>(12);
    uint32_t runtime_arg13 = get_arg_val<uint32_t>(13);
    uint32_t runtime_arg14 = get_arg_val<uint32_t>(14);
    uint32_t runtime_arg15 = get_arg_val<uint32_t>(15);
    uint32_t runtime_arg16 = get_arg_val<uint32_t>(16);
    uint32_t runtime_arg17 = get_arg_val<uint32_t>(17);
    uint32_t runtime_arg18 = get_arg_val<uint32_t>(18);
    uint32_t runtime_arg19 = get_arg_val<uint32_t>(19);
    uint32_t runtime_arg20 = get_arg_val<uint32_t>(20);
    uint32_t runtime_arg21 = get_arg_val<uint32_t>(21);
    uint32_t runtime_arg22 = get_arg_val<uint32_t>(22);
    uint32_t runtime_arg23 = get_arg_val<uint32_t>(23);
    uint32_t runtime_arg24 = get_arg_val<uint32_t>(24);
    uint32_t runtime_arg25 = get_arg_val<uint32_t>(25);
    uint32_t runtime_arg26 = get_arg_val<uint32_t>(26);
    uint32_t runtime_arg27 = get_arg_val<uint32_t>(27);
    uint32_t runtime_arg28 = get_arg_val<uint32_t>(28);
    uint32_t runtime_arg29 = get_arg_val<uint32_t>(29);
    uint32_t runtime_arg30 = get_arg_val<uint32_t>(30);
    uint32_t runtime_arg31 = get_arg_val<uint32_t>(31);
    uint32_t runtime_arg32 = get_arg_val<uint32_t>(32);
    uint32_t runtime_arg33 = get_arg_val<uint32_t>(33);
    uint32_t runtime_arg34 = get_arg_val<uint32_t>(34);
    uint32_t runtime_arg35 = get_arg_val<uint32_t>(35);
    uint32_t runtime_arg36 = get_arg_val<uint32_t>(36);
    uint32_t runtime_arg37 = get_arg_val<uint32_t>(37);
    uint32_t runtime_arg38 = get_arg_val<uint32_t>(38);
    uint32_t runtime_arg39 = get_arg_val<uint32_t>(39);
    uint32_t runtime_arg40 = get_arg_val<uint32_t>(40);
    uint32_t runtime_arg41 = get_arg_val<uint32_t>(41);
    uint32_t runtime_arg42 = get_arg_val<uint32_t>(42);
    uint32_t runtime_arg43 = get_arg_val<uint32_t>(43);
    uint32_t runtime_arg44 = get_arg_val<uint32_t>(44);
    uint32_t runtime_arg45 = get_arg_val<uint32_t>(45);
    uint32_t runtime_arg46 = get_arg_val<uint32_t>(46);
    uint32_t runtime_arg47 = get_arg_val<uint32_t>(47);
    uint32_t runtime_arg48 = get_arg_val<uint32_t>(48);
    uint32_t runtime_arg49 = get_arg_val<uint32_t>(49);
    uint32_t runtime_arg50 = get_arg_val<uint32_t>(50);
    uint32_t runtime_arg51 = get_arg_val<uint32_t>(51);
    uint32_t runtime_arg52 = get_arg_val<uint32_t>(52);
    uint32_t runtime_arg53 = get_arg_val<uint32_t>(53);
    uint32_t runtime_arg54 = get_arg_val<uint32_t>(54);
    uint32_t runtime_arg55 = get_arg_val<uint32_t>(55);
    uint32_t runtime_arg56 = get_arg_val<uint32_t>(56);
    uint32_t runtime_arg57 = get_arg_val<uint32_t>(57);
    uint32_t runtime_arg58 = get_arg_val<uint32_t>(58);
    uint32_t runtime_arg59 = get_arg_val<uint32_t>(59);
    uint32_t runtime_arg60 = get_arg_val<uint32_t>(60);
    uint32_t runtime_arg61 = get_arg_val<uint32_t>(61);
    uint32_t runtime_arg62 = get_arg_val<uint32_t>(62);
    uint32_t runtime_arg63 = get_arg_val<uint32_t>(63);
    uint32_t runtime_arg64 = get_arg_val<uint32_t>(64);
    uint32_t runtime_arg65 = get_arg_val<uint32_t>(65);
    uint32_t runtime_arg66 = get_arg_val<uint32_t>(66);
    uint32_t runtime_arg67 = get_arg_val<uint32_t>(67);
    uint32_t runtime_arg68 = get_arg_val<uint32_t>(68);
    uint32_t runtime_arg69 = get_arg_val<uint32_t>(69);
    uint32_t runtime_arg70 = get_arg_val<uint32_t>(70);
    uint32_t runtime_arg71 = get_arg_val<uint32_t>(71);
    uint32_t runtime_arg72 = get_arg_val<uint32_t>(72);
    uint32_t runtime_arg73 = get_arg_val<uint32_t>(73);
    uint32_t runtime_arg74 = get_arg_val<uint32_t>(74);
    uint32_t runtime_arg75 = get_arg_val<uint32_t>(75);
    uint32_t runtime_arg76 = get_arg_val<uint32_t>(76);
    uint32_t runtime_arg77 = get_arg_val<uint32_t>(77);
    uint32_t runtime_arg78 = get_arg_val<uint32_t>(78);
    uint32_t runtime_arg79 = get_arg_val<uint32_t>(79);
    uint32_t runtime_arg80 = get_arg_val<uint32_t>(80);
    uint32_t runtime_arg81 = get_arg_val<uint32_t>(81);
    uint32_t runtime_arg82 = get_arg_val<uint32_t>(82);
    uint32_t runtime_arg83 = get_arg_val<uint32_t>(83);
    uint32_t runtime_arg84 = get_arg_val<uint32_t>(84);
    uint32_t runtime_arg85 = get_arg_val<uint32_t>(85);
    uint32_t runtime_arg86 = get_arg_val<uint32_t>(86);
    uint32_t runtime_arg87 = get_arg_val<uint32_t>(87);
    uint32_t runtime_arg88 = get_arg_val<uint32_t>(88);
    uint32_t runtime_arg89 = get_arg_val<uint32_t>(89);
    uint32_t runtime_arg90 = get_arg_val<uint32_t>(90);
    uint32_t runtime_arg91 = get_arg_val<uint32_t>(91);
    uint32_t runtime_arg92 = get_arg_val<uint32_t>(92);
    uint32_t runtime_arg93 = get_arg_val<uint32_t>(93);
    uint32_t runtime_arg94 = get_arg_val<uint32_t>(94);
    uint32_t runtime_arg95 = get_arg_val<uint32_t>(95);
    uint32_t runtime_arg96 = get_arg_val<uint32_t>(96);
    uint32_t runtime_arg97 = get_arg_val<uint32_t>(97);
    uint32_t runtime_arg98 = get_arg_val<uint32_t>(98);
    uint32_t runtime_arg99 = get_arg_val<uint32_t>(99);
    uint32_t runtime_arg100 = get_arg_val<uint32_t>(100);
    uint32_t runtime_arg101 = get_arg_val<uint32_t>(101);
    uint32_t runtime_arg102 = get_arg_val<uint32_t>(102);
    uint32_t runtime_arg103 = get_arg_val<uint32_t>(103);
    uint32_t runtime_arg104 = get_arg_val<uint32_t>(104);
    uint32_t runtime_arg105 = get_arg_val<uint32_t>(105);
    uint32_t runtime_arg106 = get_arg_val<uint32_t>(106);
    uint32_t runtime_arg107 = get_arg_val<uint32_t>(107);
    uint32_t runtime_arg108 = get_arg_val<uint32_t>(108);
    uint32_t runtime_arg109 = get_arg_val<uint32_t>(109);
    uint32_t runtime_arg110 = get_arg_val<uint32_t>(110);
    uint32_t runtime_arg111 = get_arg_val<uint32_t>(111);
    uint32_t runtime_arg112 = get_arg_val<uint32_t>(112);
    uint32_t runtime_arg113 = get_arg_val<uint32_t>(113);
    uint32_t runtime_arg114 = get_arg_val<uint32_t>(114);
    uint32_t runtime_arg115 = get_arg_val<uint32_t>(115);
    uint32_t runtime_arg116 = get_arg_val<uint32_t>(116);
    uint32_t runtime_arg117 = get_arg_val<uint32_t>(117);
    uint32_t runtime_arg118 = get_arg_val<uint32_t>(118);
    uint32_t runtime_arg119 = get_arg_val<uint32_t>(119);
    uint32_t runtime_arg120 = get_arg_val<uint32_t>(120);
    uint32_t runtime_arg121 = get_arg_val<uint32_t>(121);
    uint32_t runtime_arg122 = get_arg_val<uint32_t>(122);
    uint32_t runtime_arg123 = get_arg_val<uint32_t>(123);
    uint32_t runtime_arg124 = get_arg_val<uint32_t>(124);
    uint32_t runtime_arg125 = get_arg_val<uint32_t>(125);
    uint32_t runtime_arg126 = get_arg_val<uint32_t>(126);
    uint32_t runtime_arg127 = get_arg_val<uint32_t>(127);
    uint32_t runtime_arg128 = get_arg_val<uint32_t>(128);
    uint32_t runtime_arg129 = get_arg_val<uint32_t>(129);
    uint32_t runtime_arg130 = get_arg_val<uint32_t>(130);
    uint32_t runtime_arg131 = get_arg_val<uint32_t>(131);
    uint32_t runtime_arg132 = get_arg_val<uint32_t>(132);
    uint32_t runtime_arg133 = get_arg_val<uint32_t>(133);
    uint32_t runtime_arg134 = get_arg_val<uint32_t>(134);
    uint32_t runtime_arg135 = get_arg_val<uint32_t>(135);
    uint32_t runtime_arg136 = get_arg_val<uint32_t>(136);
    uint32_t runtime_arg137 = get_arg_val<uint32_t>(137);
    uint32_t runtime_arg138 = get_arg_val<uint32_t>(138);
    uint32_t runtime_arg139 = get_arg_val<uint32_t>(139);
    uint32_t runtime_arg140 = get_arg_val<uint32_t>(140);
    uint32_t runtime_arg141 = get_arg_val<uint32_t>(141);
    uint32_t runtime_arg142 = get_arg_val<uint32_t>(142);
    uint32_t runtime_arg143 = get_arg_val<uint32_t>(143);
    uint32_t runtime_arg144 = get_arg_val<uint32_t>(144);
    uint32_t runtime_arg145 = get_arg_val<uint32_t>(145);
    uint32_t runtime_arg146 = get_arg_val<uint32_t>(146);
    uint32_t runtime_arg147 = get_arg_val<uint32_t>(147);
    uint32_t runtime_arg148 = get_arg_val<uint32_t>(148);
    uint32_t runtime_arg149 = get_arg_val<uint32_t>(149);
    uint32_t runtime_arg150 = get_arg_val<uint32_t>(150);
    uint32_t runtime_arg151 = get_arg_val<uint32_t>(151);
    uint32_t runtime_arg152 = get_arg_val<uint32_t>(152);
    uint32_t runtime_arg153 = get_arg_val<uint32_t>(153);
    uint32_t runtime_arg154 = get_arg_val<uint32_t>(154);
    uint32_t runtime_arg155 = get_arg_val<uint32_t>(155);
    uint32_t runtime_arg156 = get_arg_val<uint32_t>(156);
    uint32_t runtime_arg157 = get_arg_val<uint32_t>(157);
    uint32_t runtime_arg158 = get_arg_val<uint32_t>(158);
    uint32_t runtime_arg159 = get_arg_val<uint32_t>(159);
    uint32_t runtime_arg160 = get_arg_val<uint32_t>(160);
    uint32_t runtime_arg161 = get_arg_val<uint32_t>(161);
    uint32_t runtime_arg162 = get_arg_val<uint32_t>(162);
    uint32_t runtime_arg163 = get_arg_val<uint32_t>(163);
    uint32_t runtime_arg164 = get_arg_val<uint32_t>(164);
    uint32_t runtime_arg165 = get_arg_val<uint32_t>(165);
    uint32_t runtime_arg166 = get_arg_val<uint32_t>(166);
    uint32_t runtime_arg167 = get_arg_val<uint32_t>(167);
    uint32_t runtime_arg168 = get_arg_val<uint32_t>(168);
    uint32_t runtime_arg169 = get_arg_val<uint32_t>(169);
    uint32_t runtime_arg170 = get_arg_val<uint32_t>(170);
    uint32_t runtime_arg171 = get_arg_val<uint32_t>(171);
    uint32_t runtime_arg172 = get_arg_val<uint32_t>(172);
    uint32_t runtime_arg173 = get_arg_val<uint32_t>(173);
    uint32_t runtime_arg174 = get_arg_val<uint32_t>(174);
    uint32_t runtime_arg175 = get_arg_val<uint32_t>(175);
    uint32_t runtime_arg176 = get_arg_val<uint32_t>(176);
    uint32_t runtime_arg177 = get_arg_val<uint32_t>(177);
    uint32_t runtime_arg178 = get_arg_val<uint32_t>(178);
    uint32_t runtime_arg179 = get_arg_val<uint32_t>(179);
    uint32_t runtime_arg180 = get_arg_val<uint32_t>(180);
    uint32_t runtime_arg181 = get_arg_val<uint32_t>(181);
    uint32_t runtime_arg182 = get_arg_val<uint32_t>(182);
    uint32_t runtime_arg183 = get_arg_val<uint32_t>(183);
    uint32_t runtime_arg184 = get_arg_val<uint32_t>(184);
    uint32_t runtime_arg185 = get_arg_val<uint32_t>(185);
    uint32_t runtime_arg186 = get_arg_val<uint32_t>(186);
    uint32_t runtime_arg187 = get_arg_val<uint32_t>(187);
    uint32_t runtime_arg188 = get_arg_val<uint32_t>(188);
    uint32_t runtime_arg189 = get_arg_val<uint32_t>(189);
    uint32_t runtime_arg190 = get_arg_val<uint32_t>(190);
    uint32_t runtime_arg191 = get_arg_val<uint32_t>(191);
    uint32_t runtime_arg192 = get_arg_val<uint32_t>(192);
    uint32_t runtime_arg193 = get_arg_val<uint32_t>(193);
    uint32_t runtime_arg194 = get_arg_val<uint32_t>(194);
    uint32_t runtime_arg195 = get_arg_val<uint32_t>(195);
    uint32_t runtime_arg196 = get_arg_val<uint32_t>(196);
    uint32_t runtime_arg197 = get_arg_val<uint32_t>(197);
    uint32_t runtime_arg198 = get_arg_val<uint32_t>(198);
    uint32_t runtime_arg199 = get_arg_val<uint32_t>(199);
    uint32_t runtime_arg200 = get_arg_val<uint32_t>(200);
    uint32_t runtime_arg201 = get_arg_val<uint32_t>(201);
    uint32_t runtime_arg202 = get_arg_val<uint32_t>(202);
    uint32_t runtime_arg203 = get_arg_val<uint32_t>(203);
    uint32_t runtime_arg204 = get_arg_val<uint32_t>(204);
    uint32_t runtime_arg205 = get_arg_val<uint32_t>(205);
    uint32_t runtime_arg206 = get_arg_val<uint32_t>(206);
    uint32_t runtime_arg207 = get_arg_val<uint32_t>(207);
    uint32_t runtime_arg208 = get_arg_val<uint32_t>(208);
    uint32_t runtime_arg209 = get_arg_val<uint32_t>(209);
    uint32_t runtime_arg210 = get_arg_val<uint32_t>(210);
    uint32_t runtime_arg211 = get_arg_val<uint32_t>(211);
    uint32_t runtime_arg212 = get_arg_val<uint32_t>(212);
    uint32_t runtime_arg213 = get_arg_val<uint32_t>(213);
    uint32_t runtime_arg214 = get_arg_val<uint32_t>(214);
    uint32_t runtime_arg215 = get_arg_val<uint32_t>(215);
    uint32_t runtime_arg216 = get_arg_val<uint32_t>(216);
    uint32_t runtime_arg217 = get_arg_val<uint32_t>(217);
    uint32_t runtime_arg218 = get_arg_val<uint32_t>(218);
    uint32_t runtime_arg219 = get_arg_val<uint32_t>(219);
    uint32_t runtime_arg220 = get_arg_val<uint32_t>(220);
    uint32_t runtime_arg221 = get_arg_val<uint32_t>(221);
    uint32_t runtime_arg222 = get_arg_val<uint32_t>(222);
    uint32_t runtime_arg223 = get_arg_val<uint32_t>(223);
    uint32_t runtime_arg224 = get_arg_val<uint32_t>(224);
    uint32_t runtime_arg225 = get_arg_val<uint32_t>(225);
    uint32_t runtime_arg226 = get_arg_val<uint32_t>(226);
    uint32_t runtime_arg227 = get_arg_val<uint32_t>(227);
    uint32_t runtime_arg228 = get_arg_val<uint32_t>(228);
    uint32_t runtime_arg229 = get_arg_val<uint32_t>(229);
    uint32_t runtime_arg230 = get_arg_val<uint32_t>(230);
    uint32_t runtime_arg231 = get_arg_val<uint32_t>(231);
    uint32_t runtime_arg232 = get_arg_val<uint32_t>(232);
    uint32_t runtime_arg233 = get_arg_val<uint32_t>(233);
    uint32_t runtime_arg234 = get_arg_val<uint32_t>(234);
    uint32_t runtime_arg235 = get_arg_val<uint32_t>(235);
    uint32_t runtime_arg236 = get_arg_val<uint32_t>(236);
    uint32_t runtime_arg237 = get_arg_val<uint32_t>(237);
    uint32_t runtime_arg238 = get_arg_val<uint32_t>(238);
    uint32_t runtime_arg239 = get_arg_val<uint32_t>(239);
    uint32_t runtime_arg240 = get_arg_val<uint32_t>(240);
    uint32_t runtime_arg241 = get_arg_val<uint32_t>(241);
    uint32_t runtime_arg242 = get_arg_val<uint32_t>(242);
    uint32_t runtime_arg243 = get_arg_val<uint32_t>(243);
    uint32_t runtime_arg244 = get_arg_val<uint32_t>(244);
    uint32_t runtime_arg245 = get_arg_val<uint32_t>(245);
    uint32_t runtime_arg246 = get_arg_val<uint32_t>(246);
    uint32_t runtime_arg247 = get_arg_val<uint32_t>(247);
    uint32_t runtime_arg248 = get_arg_val<uint32_t>(248);
    uint32_t runtime_arg249 = get_arg_val<uint32_t>(249);
    uint32_t runtime_arg250 = get_arg_val<uint32_t>(250);
    uint32_t runtime_arg251 = get_arg_val<uint32_t>(251);
    uint32_t runtime_arg252 = get_arg_val<uint32_t>(252);
    uint32_t runtime_arg253 = get_arg_val<uint32_t>(253);
    uint32_t runtime_arg254 = get_arg_val<uint32_t>(254);
}